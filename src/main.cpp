#if 0

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

void test_parse(std::string const &filename)
{
    std::vector<int> dim = std::vector<int>();
    auto idx = filename.find("__");
    auto subStr = filename.substr(idx + 2);
    std::istringstream ss(subStr);
    char c;
    do
    {
        int val;
        ss >> val;
        dim.push_back(val);
    } while (ss >> c, c == 'x');
    for (auto const x : dim)
    {
        std::cout << x << ' ';
    }
    std::cout << std::endl;
}

int main()
{
    std::ifstream f("tensorNames.txt");
    std::string line;
    while (std::getline(f, line))
    {
        test_parse(line);
    }
    f.close();
    return 0;
}
#endif

#if 0

#include "includes.h"
#include "flavr.h"

std::vector<VidFrame> grab_vid_frames(std::string f_name)
{
    return std::vector<VidFrame>(); // TODO
}

int num_frames_new(int num_frames_orig, int expansion_coeff)
{
    return 0; // TODO
}

int main(int argc, char *argv[])
{
    std::string in_file = "";  // TODO: parsy parsy
    std::string out_file = ""; // TODO: parsy parsy
    // number of frames to add between every 2 source frames
    int expansion_coeff = 0;
    // TODO: get the model/interpolation factor
    // TODO: everything exists and valid yada yada

    // TODO: Load Model params
    // TODO: GPU side model
    // TODO: memcpys???
    auto cuFlavr = Flavr();

    std::cout << "Done Loading Network" << std::endl;

    // TODO: load input video
    auto orig_frames = grab_vid_frames(in_file);
    // TODO: make output structure
    auto output_size = num_frames_new(orig_frames.size(), expansion_coeff);
    // auto result_frames = std::vector<VidFrame>(output_size);
    std::vector<VidFrame> result_frames;
    result_frames.reserve(output_size);

    // TODO: sliding window of 4 frames feed to model, get back result, add to output
    //  TODO: what do about first couple and last couple frames??
    for (int i = 0; i < orig_frames.size(); i++)
    {
        //! Will error as bounds are wrong
        auto input_packet_start = &(orig_frames.data()[i]);
        auto guessed_frames = cuFlavr.runModel(input_packet_start);
        result_frames.insert(result_frames.end(), guessed_frames.begin(), guessed_frames.end());
        // TODO progress bar
    }

    // TODO: Write output video to file
    // TODO: cleanup

    return 0;
}

#endif
