
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<chrono>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <curand_kernel.h>
#include "curand.h"

using namespace std;

const int matsize = 2000;

__global__
void multiply(float* m1, float* m2, float* res)
{
	for (int i = threadIdx.x * 2; i < (threadIdx.x + 1) * 2; ++i) {
		for (int j = 0; j < matsize; ++j) {
			float sum = 0;
			for (int k = 0; k < matsize; ++k) {
				sum += (m1[i * matsize + k] * m2[k * matsize + j]);
			}
			res[i * matsize + j] = sum;
		}
	}
}

__global__
void randomMatrix(float* m1, float* m2) {
	curandState st;
	int id = threadIdx.x;
	curand_init(id, id, 0, &st);
	for (int i = threadIdx.x * 2; i < (threadIdx.x + 1) * 2; ++i) {
		for (int j = 0; j < matsize; ++j) {
			float f = curand_uniform(&st);
			m1[i * matsize + j] = f;
		}
	}
	for (int i = threadIdx.x * 2; i < (threadIdx.x + 1) * 2; ++i) {
		for (int j = 0; j < matsize; ++j) {
			float f = curand_uniform(&st);
			m2[i * matsize + j] = f;
		}
	}
}
void dispose(float* m1, float* m2, float* res) {
	cudaFree(m1);
	cudaFree(m2);
	cudaFree(res);
}
int multiplyWithCuda(float* m1, float* m2, float* res) {
	float* dev_m1;
	float* dev_m2;
	float* dev_res;

	m1[12] = 12;


	cudaError_t err;

	if (err = cudaMalloc((void**)&dev_m1, sizeof(float) * matsize * matsize)) {
		cout << "malloc m1 error" << endl;
		dispose(dev_m1, dev_m2, dev_res);
		return 1;
	}
	if (err = cudaMalloc((void**)&dev_m2, sizeof(float) * matsize * matsize)) {
		cout << "malloc m2 error" << endl;
		dispose(dev_m1, dev_m2, dev_res);
		return 1;
	}
	if (err = cudaMalloc((void**)&dev_res, sizeof(float) * matsize * matsize)) {
		cout << "malloc res error" << endl;
		dispose(dev_m1, dev_m2, dev_res);
		return 1;
	}
	randomMatrix << <1, 1000 >> > (dev_m1, dev_m2);
	if (err = cudaDeviceSynchronize()) {
		cout << "cant sync" << endl;
		dispose(dev_m1, dev_m2, dev_res);
		return 1;
	}
	multiply<<<1,1000>>>(dev_m1, dev_m2, dev_res);
	if (err = cudaMemcpy(m1, dev_m1, matsize * matsize * sizeof(float), cudaMemcpyDeviceToHost)) {
		cout << "copy m1 error" << endl;
		dispose(dev_m1, dev_m2, dev_res);
		return 1;
	}
	if (err = cudaMemcpy(m2, dev_m2, matsize * matsize * sizeof(float), cudaMemcpyDeviceToHost)) {
		cout << "copy m2 error" << endl;
		dispose(dev_m1, dev_m2, dev_res);
		return 1;
	}
	if (err = cudaMemcpy(res, dev_res, matsize * matsize * sizeof(float), cudaMemcpyDeviceToHost)) {
		cout << "copy res error" << endl;
		dispose(dev_m1, dev_m2, dev_res);
		return 1;
	}
	dispose(dev_m1, dev_m2, dev_res);

	return 0;
}

void report(chrono::milliseconds t) {
	fstream ofs = fstream("output.txt", ios_base::out);
	stringstream ss;
	ss << "matrix size: " << matsize <<" x " <<matsize<< endl;
	ss << "total time: " << (float)t.count() / 1000 << endl;
	ofs << ss.str();
	ofs.flush();
	ofs.close();
}
int main() {
	float* m1 = new float[matsize * matsize];
	float* m2 = new float[matsize * matsize];
	float* res = new float[matsize * matsize];

	cudaSetDevice(0);
	auto start = chrono::system_clock::now();
	multiplyWithCuda(m1, m2, res);
	auto dur = chrono::duration_cast<chrono::milliseconds> (chrono::system_clock::now() - start);

	report(dur);
	return 0;
}