#pragma once

#include "CxException.hpp"
#include <Conversion.hpp>

/**
 * A class to handle non-interleaved AVI (and possibly other formats) by remuxing
 * in-memory. If no remux is needed, we just open the file normally (fewer overhead).
 */
class Remuxer
{
  public:
    /**
     * Construct a Remuxer for a given file path.
     * The actual open or remux process is done in prepare().
     */
    Remuxer(const std::string& filePath);

    /**
     * Destructor will free any in-memory buffers, and close the AVFormatContext if
     * opened.
     */
    ~Remuxer();

    /**
     * The main entry point:
     *  - Checks if the file is non-interleaved AVI.
     *  - If so, remuxes in memory.
     *  - Opens the final data (memory or file) in an AVFormatContext.
     * Returns a pointer to the AVFormatContext that you can use for decoding.
     * On error, returns nullptr.
     *
     * Ownership note: The Remuxer keeps ownership of the context. Don't close it
     * yourself. Once the Remuxer is destroyed, it closes everything automatically.
     */
    AVFormatContext* prepare();

  private:
    /**
     * Detect if the file is an AVI and if it is flagged as non-interleaved by FFmpeg.
     */
    bool isNonInterleavedAvi(const std::string& filePath);

    /**
     * Remux the input file in memory (packet by packet).
     * On success, we store the resulting memory buffer (bufferData_, bufferSize_).
     * Return 0 on success, negative on error.
     */
    int remuxInMemory(const std::string& inPath);

    /**
     * Once we have bufferData_, bufferSize_, open a new AVFormatContext from that
     * buffer. Returns a pointer on success, or nullptr on error.
     */
    AVFormatContext* openFromMemoryBuffer();

    /**
     * Freed automatically in destructor, but you can call it in error paths if needed.
     */
    void freeMemoryResources();

  private:
    std::string filePath_;        ///< The original file path
    bool triedToPrepare_ = false; ///< So we only do the process once

    AVFormatContext* formatCtx_ =
        nullptr; ///< The final context returned from prepare()

    // For the in-memory buffer approach:
    unsigned char* bufferData_ = nullptr; ///< Allocated by avio_open_dyn_buf()
    size_t bufferSize_ = 0;               ///< Size of bufferData_

    // We might store info about whether we used memory remux or not
    bool usedMemoryRemux_ = false;
};

