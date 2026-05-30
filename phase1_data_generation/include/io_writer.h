#pragma once
#include "blocking_queue.h"
#include "puzzle_state.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// AsyncWriter — dedicated I/O thread that drains the result queue and batch-
// writes the dataset to disk in a compact binary format.
//
// File format (little-endian):
//   Header (16 bytes):
//     [0..3]  uint32 magic   = 0x50313544 ("P15D")
//     [4..7]  uint32 version = 1
//     [8..15] uint64 n_records   (patched on close)
//   Record (9 bytes):
//     [0..7]  uint64 state   (nibble-packed board)
//     [8]     uint8  cost    (optimal cost-to-go)
//   Files roll at `max_file_bytes` -> dataset_000.bin, dataset_001.bin, ...
// ---------------------------------------------------------------------------
namespace puzzle {

struct SolvedRecord {
    State   state;
    uint8_t cost;
};

class AsyncWriter {
public:
    static constexpr uint32_t MAGIC        = 0x50313544u;
    static constexpr uint32_t VERSION      = 1u;
    static constexpr int      RECORD_BYTES = 9;
    static constexpr int      HEADER_BYTES = 16;

    explicit AsyncWriter(std::string output_dir,
                         std::size_t max_file_bytes = 1ULL << 30)
        : output_dir_(std::move(output_dir)), max_file_bytes_(max_file_bytes) {}

    void start(BlockingQueue<SolvedRecord>& result_queue) {
        thread_ = std::thread(&AsyncWriter::run, this, std::ref(result_queue));
    }

    void stop() {
        if (thread_.joinable()) thread_.join();
    }

    uint64_t records_written() const noexcept { return records_written_.load(); }

private:
    void run(BlockingQueue<SolvedRecord>& q) {
        std::error_code ec;
        std::filesystem::create_directories(output_dir_, ec);

        std::vector<SolvedRecord> batch;
        batch.reserve(BATCH);
        open_new_file();

        SolvedRecord rec;
        while (q.pop(rec)) {
            batch.push_back(rec);
            if (batch.size() >= BATCH) { write_batch(batch); batch.clear(); }
        }
        if (!batch.empty()) write_batch(batch);
        close_file();
    }

    void open_new_file() {
        std::ostringstream ss;
        ss << output_dir_ << "/dataset_"
           << std::setw(3) << std::setfill('0') << file_index_++ << ".bin";
        current_path_ = ss.str();

        file_.open(current_path_, std::ios::binary);
        if (!file_) throw std::runtime_error("Cannot open output file: " + current_path_);

        const uint32_t magic = MAGIC, version = VERSION;
        const uint64_t n_rec = 0;
        file_.write(reinterpret_cast<const char*>(&magic), 4);
        file_.write(reinterpret_cast<const char*>(&version), 4);
        file_.write(reinterpret_cast<const char*>(&n_rec), 8);

        current_file_records_ = 0;
        current_file_bytes_   = HEADER_BYTES;
    }

    void close_file() {
        if (!file_.is_open()) return;
        file_.seekp(8); // n_records field
        const uint64_t n = current_file_records_;
        file_.write(reinterpret_cast<const char*>(&n), 8);
        file_.close();
        std::cout << "[IO] Closed " << current_path_
                  << " (" << current_file_records_ << " records)\n" << std::flush;
    }

    void write_batch(const std::vector<SolvedRecord>& batch) {
        for (const auto& r : batch) {
            if (current_file_bytes_ + RECORD_BYTES > static_cast<long long>(max_file_bytes_)) {
                close_file();
                open_new_file();
            }
            file_.write(reinterpret_cast<const char*>(&r.state), 8);
            file_.write(reinterpret_cast<const char*>(&r.cost), 1);
            ++current_file_records_;
            current_file_bytes_ += RECORD_BYTES;
        }
        records_written_.fetch_add(batch.size(), std::memory_order_relaxed);
    }

    static constexpr std::size_t BATCH = 8192;

    std::string           output_dir_;
    std::size_t           max_file_bytes_;
    std::thread           thread_;
    std::atomic<uint64_t> records_written_{0};

    std::ofstream file_;
    std::string   current_path_;
    int           file_index_           = 0;
    uint64_t      current_file_records_ = 0;
    long long     current_file_bytes_   = 0;
};

} // namespace puzzle
