#include "vectors.h"

#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>

namespace vkp_test {

std::vector<Vector> load_all(const std::string &dir) {
    std::vector<std::string> names;
    DIR *d = opendir(dir.c_str());
    if (!d) return {};
    while (struct dirent *e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        std::string sub = dir + "/" + e->d_name;
        struct stat st;
        if (stat(sub.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) names.emplace_back(e->d_name);
    }
    closedir(d);
    std::sort(names.begin(), names.end());

    std::vector<Vector> out;
    for (const auto &n : names) {
        Vector v;
        if (parse_vector(dir + "/" + n + "/data.yaml", n, v)) out.push_back(std::move(v));
    }
    return out;
}

} // namespace vkp_test
