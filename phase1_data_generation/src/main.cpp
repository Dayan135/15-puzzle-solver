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
// The heuristic is held read-only and shared across workers.  This skeleton
// uses ManhattanHeuristic; once `build_pdbs` exists, load the additive PDBs
// into a SumHeuristic and instantiate IDAStar<SumHeuristic> instead (the
// solver is templated, so it is a one-line swap).
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

struct Args {
    int         threads      = std::max(1u, std::thread::hardware_concurrency());
    uint64_t    target       = 1000;
    int         scramble_len = 1000;     // random moves applied to GOAL
    std::string out_dir      = "./data";
    uint64_t    seed         = 0xC0FFEE;
    std::size_t queue_cap    = 1u << 14;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* def) -> std::string {
            return (i + 1 < argc) ? argv[++i] : def;
        };
        if      (!std::strcmp(argv[i], "--threads"))  a.threads      = std::stoi(next("1"));
        else if (!std::strcmp(argv[i], "--target"))   a.target       = std::stoull(next("0"));
        else if (!std::strcmp(argv[i], "--scramble")) a.scramble_len = std::stoi(next("1000"));
        else if (!std::strcmp(argv[i], "--out-dir"))  a.out_dir      = next("./data");
        else if (!std::strcmp(argv[i], "--seed"))     a.seed         = std::stoull(next("0"));
        else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
        }
    }
    if (a.threads < 1) a.threads = 1;
    return a;
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    std::cout << "[gen] threads=" << args.threads
              << " target=" << args.target
              << " scramble=" << args.scramble_len
              << " out=" << args.out_dir
              << " heuristic=Manhattan(placeholder)\n";

    const ManhattanHeuristic heur;  // shared, read-only

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
            IDAStar<ManhattanHeuristic> solver(heur);
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

    // --- generator (this thread) ---
    const auto t0 = std::chrono::steady_clock::now();
    std::mt19937_64 rng(args.seed);
    for (uint64_t i = 0; i < args.target; ++i) {
        work_q.push(scramble(GOAL, args.scramble_len, rng));
        if ((i + 1) % 100000 == 0)
            std::cout << "[gen] queued " << (i + 1) << " / " << args.target << "\n";
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
