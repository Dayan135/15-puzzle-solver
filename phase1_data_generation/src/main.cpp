// main.cpp — Data generation orchestrator
//
// Usage:
//   generate_data [options]
//   Options:
//     --threads N     Number of solver threads (default: 14)
//     --target  M     Total records to generate (default: 100000000)
//     --pdb-dir P     Directory containing *.pdb files (default: .)
//     --out-dir O     Output directory for dataset_*.bin files (default: .)

#include "cube_state.h"
#include "move_tables.h"
#include "pdb.h"
#include "ida_star.h"
#include "blocking_queue.h"
#include "io_writer.h"

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <string>
#include <cstring>
#include <iomanip>
#include <filesystem>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
struct Config {
    int         threads  = 14;
    uint64_t    target   = 100'000'000ULL;
    std::string pdb_dir  = ".";
    std::string out_dir  = ".";
};

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc - 1; ++i) {
        std::string key = argv[i];
        std::string val = argv[i + 1];
        if (key == "--threads") { cfg.threads  = std::stoi(val);   ++i; }
        else if (key == "--target")  { cfg.target   = std::stoull(val); ++i; }
        else if (key == "--pdb-dir") { cfg.pdb_dir  = val;             ++i; }
        else if (key == "--out-dir") { cfg.out_dir  = val;             ++i; }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Solver thread function
// ---------------------------------------------------------------------------
static void solver_thread(
    BlockingQueue<CubeState>&    work_q,
    BlockingQueue<SolvedRecord>& result_q,
    const MoveTables&            mt,
    const MaxHeuristic&          heuristic,
    std::atomic<uint64_t>&       total_solved)
{
    IDAStar solver(mt, heuristic);
    CubeState state;
    while (work_q.pop(state)) {
        uint8_t cost = solver.solve(state);
        result_q.push(SolvedRecord{state, cost});
        total_solved.fetch_add(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Progress monitor (runs on the main thread between state-generation bursts)
// ---------------------------------------------------------------------------
static void print_progress(uint64_t done, uint64_t target,
                            uint64_t elapsed_sec,
                            const uint64_t cost_hist[21]) {
    double pct  = 100.0 * done / target;
    double rate = elapsed_sec > 0 ? static_cast<double>(done) / elapsed_sec : 0.0;
    uint64_t eta_sec = (rate > 0 && done < target)
                     ? static_cast<uint64_t>((target - done) / rate)
                     : 0;

    std::cout << "\r[Progress] "
              << done << "/" << target
              << " (" << std::fixed << std::setprecision(1) << pct << "%)"
              << "  rate=" << std::setprecision(1) << rate << "/s"
              << "  ETA=" << eta_sec / 3600 << "h"
              << std::setw(2) << std::setfill('0') << (eta_sec % 3600) / 60 << "m"
              << "  " << std::flush;
    (void)cost_hist; // histogram printed in separate periodic reports
}

static void print_histogram(const uint64_t cost_hist[21], uint64_t total) {
    std::cout << "\n--- Cost distribution (last snapshot) ---\n";
    for (int d = 0; d <= 20; ++d) {
        if (cost_hist[d] == 0) continue;
        double pct = 100.0 * cost_hist[d] / total;
        std::cout << "  depth " << std::setw(2) << d << ": "
                  << std::setw(10) << cost_hist[d]
                  << "  (" << std::fixed << std::setprecision(1) << pct << "%)\n";
    }
    std::cout << std::flush;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    std::cout << "=== Rubik's Cube Data Generator ===\n"
              << "  threads : " << cfg.threads  << "\n"
              << "  target  : " << cfg.target   << " records\n"
              << "  pdb_dir : " << cfg.pdb_dir  << "\n"
              << "  out_dir : " << cfg.out_dir  << "\n\n";

    // 1. Initialise move tables (singleton, safe to call before threading)
    const MoveTables& mt = MoveTables::get();

    // 2. Load PDBs
    auto pdb_path = [&](const std::string& name) {
        return cfg.pdb_dir + "/" + name;
    };

    std::cout << "Loading PDBs...\n" << std::flush;
    PatternDB corner_pdb(CORNER_PDB_SIZE);
    PatternDB edge_a_pdb(EDGE_PDB_SIZE);
    PatternDB edge_b_pdb(EDGE_PDB_SIZE);

    try {
        corner_pdb.load(pdb_path("corner.pdb"));
        edge_a_pdb.load(pdb_path("edge_a.pdb"));
        edge_b_pdb.load(pdb_path("edge_b.pdb"));
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what()
                  << "\nRun build_pdbs --out " << cfg.pdb_dir << " first.\n";
        return 1;
    }
    std::cout << "PDBs loaded (~86 MB in RAM).\n\n" << std::flush;

    // 3. Build heuristic (shared read-only across all threads)
    MaxHeuristic heuristic(corner_pdb, edge_a_pdb, edge_b_pdb);

    // 4. Queues
    BlockingQueue<CubeState>    work_q(2000);
    BlockingQueue<SolvedRecord> result_q(5000);

    // 5. Start I/O writer thread
    AsyncWriter writer(cfg.out_dir);
    writer.start(result_q);

    // 6. Start solver threads
    std::atomic<uint64_t> total_solved{0};
    std::vector<std::thread> solvers;
    solvers.reserve(static_cast<std::size_t>(cfg.threads));
    for (int t = 0; t < cfg.threads; ++t) {
        solvers.emplace_back(solver_thread,
                             std::ref(work_q), std::ref(result_q),
                             std::cref(mt), std::cref(heuristic),
                             std::ref(total_solved));
    }

    // 7. Main thread: generate random states and feed work queue
    uint64_t cost_hist[21] = {};
    uint64_t states_generated = 0;
    uint64_t last_hist_at = 0;

    auto t_start = std::chrono::steady_clock::now();
    auto t_last_report = t_start;

    // Use per-thread RNG seeded from hardware entropy
    std::mt19937 rng(std::random_device{}());

    while (states_generated < cfg.target) {
        CubeState s = random_state(rng);
        work_q.push(s);
        ++states_generated;

        // Periodic progress report every ~5 seconds
        auto now = std::chrono::steady_clock::now();
        auto since_report = std::chrono::duration_cast<std::chrono::seconds>(
                                now - t_last_report).count();
        if (since_report >= 5) {
            uint64_t done = total_solved.load(std::memory_order_relaxed);
            auto elapsed  = std::chrono::duration_cast<std::chrono::seconds>(
                                now - t_start).count();
            print_progress(done, cfg.target,
                           static_cast<uint64_t>(elapsed), cost_hist);
            t_last_report = now;

            // Histogram every 100k solved
            if (done - last_hist_at >= 100'000) {
                print_histogram(cost_hist, done);
                last_hist_at = done;
            }
        }
    }

    // 8. Drain: wait until all generated states are solved
    // Signal solver threads to stop once the work queue is exhausted
    work_q.shutdown();
    for (auto& t : solvers) t.join();

    // Signal I/O thread
    result_q.shutdown();
    writer.stop();

    uint64_t final_count = writer.records_written();
    auto t_end  = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count();

    std::cout << "\n\n=== Done ===\n"
              << "  Records written : " << final_count << "\n"
              << "  Elapsed         : " << elapsed / 3600 << "h "
              << (elapsed % 3600) / 60 << "m " << elapsed % 60 << "s\n"
              << "  Avg rate        : "
              << (elapsed > 0 ? final_count / elapsed : 0) << " records/s\n";

    return 0;
}
