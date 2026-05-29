#pragma once
#include "cube_state.h"
#include "blocking_queue.h"

#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <fstream>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <iostream>

// One solved record: the cube state plus its optimal depth.
struct SolvedRecord {
    CubeState state;
    uint8_t   cost;
};

// ---------------------------------------------------------------------------
// AsyncWriter — dedicated I/O thread that drains a result queue and writes
// binary dataset files to disk.
//
// File format:
//   Header (16 bytes):
//     [0..3]  uint32  magic    = 0x52435542 ("RCUB")
//     [4..7]  uint32  version  = 1
//     [8..15] uint64  n_records  (patched at file close)
//   Per record (21 bytes):
//     [0..7]  corners[8]  uint8 each
//     [8..19] edges[12]   uint8 each
//     [20]    cost        uint8
// ---------------------------------------------------------------------------
class AsyncWriter {
public:
    static constexpr uint32_t MAGIC   = 0x52435542u;
    static constexpr uint32_t VERSION = 1u;
    static constexpr int      RECORD_BYTES = 21;
    static constexpr int      HEADER_BYTES = 16;

    // max_file_bytes: roll over to a new file when this size is reached.
    explicit AsyncWriter(std::string output_dir,
                         std::size_t max_file_bytes = 1ULL << 30)
        : output_dir_(std::move(output_dir))
        , max_file_bytes_(max_file_bytes)
    {}

    // Start the writer thread, consuming from result_queue.
    void start(BlockingQueue<SolvedRecord>& result_queue) {
        thread_ = std::thread(&AsyncWriter::run, this, std::ref(result_queue));
    }

    // Stop: waits for the writer thread to flush and exit.
    void stop() {
        if (thread_.joinable()) thread_.join();
    }

    uint64_t records_written() const noexcept { return records_written_.load(); }

private:
    void run(BlockingQueue<SolvedRecord>& q) {
        std::filesystem::create_directories(output_dir_);

        std::vector<SolvedRecord> batch;
        batch.reserve(4096);

        open_new_file();

        SolvedRecord rec;
        while (q.pop(rec)) {
            batch.push_back(rec);
            if (batch.size() >= 4096) {
                write_batch(batch);
                batch.clear();
            }
        }
        if (!batch.empty()) write_batch(batch);

        close_file();
    }

    void open_new_file() {
        // Build name: dataset_NNN.bin
        std::ostringstream ss;
        ss << output_dir_ << "/dataset_"
           << std::setw(3) << std::setfill('0') << file_index_
           << ".bin";
        current_path_ = ss.str();
        ++file_index_;

        file_.open(current_path_, std::ios::binary);
        if (!file_) throw std::runtime_error("Cannot open output file: " + current_path_);

        // Write header placeholder (n_records will be patched on close)
        uint32_t magic   = MAGIC;
        uint32_t version = VERSION;
        uint64_t n_rec   = 0;
        file_.write(reinterpret_cast<const char*>(&magic),   4);
        file_.write(reinterpret_cast<const char*>(&version), 4);
        file_.write(reinterpret_cast<const char*>(&n_rec),   8);

        current_file_records_ = 0;
        current_file_bytes_   = HEADER_BYTES;
    }

    void close_file() {
        if (!file_.is_open()) return;
        // Patch the record count in the header
        file_.seekp(8); // offset of n_records field
        uint64_t n = static_cast<uint64_t>(current_file_records_);
        file_.write(reinterpret_cast<const char*>(&n), 8);
        file_.close();
        std::cout << "[IO] Closed " << current_path_
                  << " (" << current_file_records_ << " records)\n" << std::flush;
    }

    void write_batch(const std::vector<SolvedRecord>& batch) {
        for (const auto& r : batch) {
            // Roll over to a new file if needed
            if (current_file_bytes_ + RECORD_BYTES > static_cast<int>(max_file_bytes_)) {
                close_file();
                open_new_file();
            }
            file_.write(reinterpret_cast<const char*>(r.state.corners), N_CORNERS);
            file_.write(reinterpret_cast<const char*>(r.state.edges),   N_EDGES);
            file_.write(reinterpret_cast<const char*>(&r.cost),         1);
            ++current_file_records_;
            current_file_bytes_ += RECORD_BYTES;
        }
        records_written_.fetch_add(batch.size(), std::memory_order_relaxed);
    }

    std::string  output_dir_;
    std::size_t  max_file_bytes_;
    std::thread  thread_;
    std::atomic<uint64_t> records_written_{0};

    std::ofstream file_;
    std::string   current_path_;
    int           file_index_           = 0;
    uint64_t      current_file_records_ = 0;
    int           current_file_bytes_   = 0;
};
