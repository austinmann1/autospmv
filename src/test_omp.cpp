#include <omp.h>
#include <iostream>
#include <vector>

int main() {
    const int n = 1000000;
    std::vector<double> a(n, 1.0);
    std::vector<double> b(n, 2.0);
    std::vector<double> c(n, 0.0);
    
    double start = omp_get_wtime();
    
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
    
    double end = omp_get_wtime();
    
    // Verify result and print time
    bool correct = true;
    for(int i = 0; i < n; i++) {
        if(c[i] != 3.0) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Time: " << (end - start) << "s\n";
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << "\n";
    std::cout << "Num threads: " << omp_get_max_threads() << "\n";
    
    return 0;
}
