// ---------------------------------------------------------------------------
// generate_data — the data-generation pipeline.
//
//   main (generator)            N worker threads             I/O thread
//   ------------------          -----------------            ----------
//   scramble GOAL  --push-->  [ BlockingQueue<State> ]
//                                  pop -> IDA* solve
//                                  push {state, cost} -->  [ BlockingQueue<SolvedRecord> ]
//                                                              pop -> dataset_NNN.bin
//
// The heuristic is held read-only and shared across workers: a SumHeuristic
// over the additive 7-8 PDBs (built once by `build_pdbs`).
//
// STRATIFIED GENERATION
//   Uniform random scrambles cluster around optimal cost 52-53 (the bell-curve
//   peak of the state space).  A network trained on that loses gradient near
//   the goal, so search plateaus expanding nodes right beside it.  Instead the
//   generator emits equal quotas from a table of scramble-length BUCKETS that
//   span the full 1..53 cost range, and cycles round-robin across them so each
//   bucket is represented evenly and workers always get a shallow/deep mix.
// ---------------------------------------------------------------------------
#include "blocking_queue.h"
#include "ida_star.h"
#include "io_writer.h"
#include "pdb.h"
#include "puzzle_state.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace puzzle;

namespace {

// Scramble-length buckets spanning the full optimal-cost range.  Shallow depths
// match scramble length exactly; deep ones saturate toward the ~52-53 peak.
const std::vector<int> DEFAULT_BUCKETS = {
    1, 2, 3, 4, 5, 6, 7, 8,
    10, 12, 15, 18, 22, 27, 33, 40,
    50, 65, 85, 110, 200, 1000
};

struct Args {
    int              threads   = std::max(1u, std::thread::hardware_concurrency());
    uint64_t         target    = 1000;
    std::string      pdb_dir   = "./data/pdbs";
    std::string      out_dir   = "./data";
    uint64_t         seed      = 0xC0FFEE;
    std::size_t      queue_cap = 1u << 14;
    std::vector<int> buckets   = DEFAULT_BUCKETS;
};

std::vector<int> parse_buckets(const std::string& csv) {
    std::vector<int> out;
    std::size_t i = 0;
    while (i < csv.size()) {
        std::size_t j = csv.find(',', i);
        if (j == std::string::npos) j = csv.size();
        if (j > i) out.push_back(std::stoi(csv.substr(i, j - i)));
        i = j + 1;
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* def) -> std::string {
            return (i + 1 < argc) ? argv[++i] : def;
        };
        if      (!std::strcmp(argv[i], "--threads"))  a.threads = std::stoi(next("1"));
        else if (!std::strcmp(argv[i], "--target"))   a.target  = std::stoull(next("0"));
        else if (!std::strcmp(argv[i], "--pdb-dir"))  a.pdb_dir = next("./data/pdbs");
        else if (!std::strcmp(argv[i], "--out-dir"))  a.out_dir = next("./data");
        else if (!std::strcmp(argv[i], "--seed"))     a.seed    = std::stoull(next("0"));
        else if (!std::strcmp(argv[i], "--buckets"))  a.buckets = parse_buckets(next(""));
        else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
        }
    }
    if (a.threads < 1) a.threads = 1;
    if (a.buckets.empty()) a.buckets = DEFAULT_BUCKETS;
    return a;
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    std::cout << "[gen] threads=" << args.threads
              << " target=" << args.target
              << " buckets=" << args.buckets.size()
              << " pdb-dir=" << args.pdb_dir
              << " out=" << args.out_dir
              << " heuristic=AdditivePDB(7-8)\n";

    SumHeuristic heur;  // shared, read-only
    if (!heur.a.load(args.pdb_dir + "/pdb_a.bin") ||
        !heur.b.load(args.pdb_dir + "/pdb_b.bin") || !heur.ready()) {
        std::cerr << "[gen] FATAL: could not load PDBs from " << args.pdb_dir
                  << " (expected pdb_a.bin, pdb_b.bin). Run build_pdbs first.\n";
        return 1;
    }

    BlockingQueue<State>        work_q(args.queue_cap);
    BlockingQueue<SolvedRecord> result_q(args.queue_cap);

    AsyncWriter writer(args.out_dir);
    writer.start(result_q);

    std::atomic<uint64_t> solved{0};
    std::atomic<int>      finished{0};
    const int             n = args.threads;

    // --- worker pool ---
    std::vector<std::thread> workers;
    workers.reserve(n);
    for (int w = 0; w < n; ++w) {
        workers.emplace_back([&] {
            IDAStar<SumHeuristic> solver(heur);
            State s;
            while (work_q.pop(s)) {
                const int cost = solver.solve(s);
                result_q.push(SolvedRecord{s, static_cast<uint8_t>(cost)});
                solved.fetch_add(1, std::memory_order_relaxed);
            }
            // Last worker out closes the result stream so the writer can drain.
            if (finished.fetch_add(1, std::memory_order_acq_rel) + 1 == n)
                result_q.shutdown();
        });
    }

    // --- generator (this thread): stratified round-robin over buckets ---
    const auto t0 = std::chrono::steady_clock::now();
    std::mt19937_64 rng(args.seed);

    std::vector<uint64_t> remaining(args.buckets.size());
    {
        const uint64_t base  = args.target / args.buckets.size();
        const uint64_t extra = args.target % args.buckets.size();
        for (std::size_t i = 0; i < args.buckets.size(); ++i)
            remaining[i] = base + (i < extra ? 1 : 0);
    }

    uint64_t queued = 0;
    bool any = true;
    while (any) {
        any = false;
        for (std::size_t i = 0; i < args.buckets.size(); ++i) {
            if (remaining[i] == 0) continue;
            work_q.push(scramble(GOAL, args.buckets[i], rng));
            --remaining[i];
            any = true;
            if (++queued % 100000 == 0)
                std::cout << "[gen] queued " << queued << " / " << args.target << "\n";
        }
    }
    work_q.shutdown();

    for (auto& t : workers) t.join();
    writer.stop();

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[gen] done: " << writer.records_written() << " records in "
              << secs << "s ("
              << (secs > 0 ? writer.records_written() / secs : 0) << " states/s)\n";
    return 0;
}
