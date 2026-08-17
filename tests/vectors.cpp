#include "vectors.h"

#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>

namespace vkzg_test {

namespace {

std::vector<std::string> list_case_dirs(const std::string &dir, const std::string &prefix) {
    std::vector<std::string> names;
    DIR *d = opendir(dir.c_str());
    if (!d) return names;
    while (struct dirent *e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        std::string name = e->d_name;
        if (name.rfind(prefix, 0) != 0) continue;
        std::string sub = dir + "/" + name;
        struct stat st;
        if (stat(sub.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) names.push_back(name);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    return names;
}

} // namespace

std::vector<Vector> load_all(const std::string &dir) {
    std::vector<Vector> out;
    for (const auto &n : list_case_dirs(dir, "compute_cells_and_kzg_proofs_case_")) {
        Vector v;
        if (parse_vector(dir + "/" + n + "/data.yaml", n, v)) out.push_back(std::move(v));
    }
    return out;
}

std::vector<RecoverVector> load_all_recover(const std::string &dir) {
    std::vector<RecoverVector> out;
    for (const auto &n : list_case_dirs(dir, "recover_cells_and_kzg_proofs_case_")) {
        RecoverVector v;
        if (parse_recover_vector(dir + "/" + n + "/data.yaml", n, v)) out.push_back(std::move(v));
    }
    return out;
}

} // namespace vkzg_test
