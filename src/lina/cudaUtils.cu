#define BLOCK_SIZE_2D 16
#define BLOCK_SIZE_1D 256

#include "utils.h"
#include "Exception.h"

__device__ float sigmoid(float z) {
	return 1.0f / (1.0f + __expf(-z));
}

__global__ void kernelSigmoidFX(unsigned int m, unsigned int n, float* src, float* dest,
		float* bias) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		float z = bias[j] + src[i * n + j];
		dest[i * n + j] = sigmoid(z);
	}
}

__global__ void kernelSigmoidDX(unsigned int m, unsigned int n, float* src, float* dest,
		float* bias) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		float z = bias[j] + src[i * n + j];
		dest[i * n + j] = sigmoid(z) * (1 - sigmoid(z));
	}
}

__global__ void fastSigmoidKernel(unsigned int m, unsigned int n, float* src, float* dest) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		float x = src[i * n + j];
		dest[i * n + j] = x / (1 + abs(x));
	}
}

__global__ void kernelTanh(unsigned int m, unsigned int n, float* src, float* dest) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		float z = src[i * n + j];
		dest[i * n + j] = tanhf(z);
	}
}

__global__ void kernelCopyMatrix(unsigned int m, unsigned int n, float* src, float* dest) {
	
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		dest[i * n + j] = src[i * n + j];
	}
}

__global__ void kernelMatrixAdd(unsigned int m, unsigned int n, float* a, float* b, float* dest) {
	
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		unsigned int absIdx = i * n + j;
		dest[absIdx] = a[absIdx] + b[absIdx];
	}
}

__global__ void kernelMatrixEleMult(unsigned int m, unsigned int n, float* a, float* b,
		float* dest) {
	
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		unsigned int absIdx = i * n + j;
		dest[absIdx] = a[absIdx] * b[absIdx];
	}
}

__global__ void kernelMatrixPow(unsigned int m, unsigned int n, float* a, float exp, float* dest) {
	
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		unsigned int absIdx = i * n + j;
		dest[absIdx] = powf(a[absIdx], exp);
	}
}

__global__ void kernelMatrixSelect(unsigned int n, unsigned int ms, unsigned int ns, unsigned int sm,
		unsigned int sn, float* src, float* dest) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < sm && j < sn) {
		unsigned int absIdx = i * sn + j;
		dest[absIdx] = src[(ms + i) * n + (ns + j)];
	}
}

__global__ void kernelCopyVector(unsigned int length, float* src, float* dest) {
	
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		dest[i] = src[i];
	}
}

__global__ void kernelMatrixInsertCol(unsigned int m, unsigned int n, unsigned int col, float val,
		float* src, float* dest) {
	
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		if (j < col) {
			dest[i * n + j] = src[i * (n - 1) + j];
		} else if (j == col) {
			dest[i * n + j] = val;
		} else {
			dest[i * n + j] = src[i * (n - 1) + j - 1];
		}
	}
}

__global__ void kernelMatrixInsertColVector(unsigned int m, unsigned int n, unsigned int col, float* vec,
		float* src, float* dest) {
	
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		if (j < col) {
			dest[i * n + j] = src[i * (n - 1) + j];
		} else if (j == col) {
			dest[i * n + j] = vec[i];
		} else {
			dest[i * n + j] = src[i * (n - 1) + j - 1];
		}
	}
}

__global__ void kernelMatrixTranspose(unsigned int m, unsigned int n, float* src, float* dest) {
	
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < n) {
		dest[j * m + i] = src[i * n + j];
	}
}

void applySigmoidFX(unsigned int m, unsigned int n, float* src, float* dest, float* bias) {
	
	dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	
	unsigned int blocksX = (unsigned int) ceil(n / (double) BLOCK_SIZE_2D);
	unsigned int blocksY = (unsigned int) ceil(m / (double) BLOCK_SIZE_2D);
	
	dim3 grid(blocksX, blocksY);
	
	kernelSigmoidFX<<<grid, block>>>(m, n, src, dest, bias);
}

void applySigmoidDX(unsigned int m, unsigned int n, float* src, float* dest, float* bias) {
	
	dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	
	unsigned int blocksX = (unsigned int) ceil(n / (double) BLOCK_SIZE_2D);
	unsigned int blocksY = (unsigned int) ceil(m / (double) BLOCK_SIZE_2D);
	
	dim3 grid(blocksX, blocksY);
	
	kernelSigmoidDX<<<grid, block>>>(m, n, src, dest, bias);
}

void matrixAdd(unsigned int m, unsigned int n, float* a, float* b, float* dest) {
	
	dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	
	unsigned int blocksX = (unsigned int) ceil(n / (double) BLOCK_SIZE_2D);
	unsigned int blocksY = (unsigned int) ceil(m / (double) BLOCK_SIZE_2D);
	
	dim3 grid(blocksX, blocksY);
	
	kernelMatrixAdd<<<grid, block>>>(m, n, a, b, dest);
}

void matrixEleMult(unsigned int m, unsigned int n, float* a, float* b, float* dest) {
	
	dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	
	unsigned int blocksX = (unsigned int) ceil(n / (double) BLOCK_SIZE_2D);
	unsigned int blocksY = (unsigned int) ceil(m / (double) BLOCK_SIZE_2D);
	
	dim3 grid(blocksX, blocksY);
	
	kernelMatrixEleMult<<<grid, block>>>(m, n, a, b, dest);
}
void matrixPow(unsigned int m, unsigned int n, float* a, float exp, float* dest) {
	
	dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	
	unsigned int blocksX = (unsigned int) ceil(n / (double) BLOCK_SIZE_2D);
	unsigned int blocksY = (unsigned int) ceil(m / (double) BLOCK_SIZE_2D);
	
	dim3 grid(blocksX, blocksY);
	
	kernelMatrixPow<<<grid, block>>>(m, n, a, exp, dest);
}

void matrixSelect(unsigned int m, unsigned int n, unsigned int ms, unsigned int ns,
		unsigned int me, unsigned int ne, float* src, float* dest) {
	
	if (ms >= me) {
		throw Exception(
				"Invalid argument ms must be lower than me. Got ms: " + to_string(ns) + ", me: "
						+ to_string(me) + ".");
	}
	
	if (ns >= ne) {
		throw Exception(
				"Invalid argument ns must be lower than ne. Got ns: " + to_string(ns) + ", ne: "
						+ to_string(ne) + ".");
	}

	if (me > m) {
		throw Exception(
				"Selection end is out of bounds. Expected <= " + to_string(m) + ", but got "
						+ to_string(me) + " instead.");
	}
	
	if (ne > n) {
		throw Exception(
				"Selection end for columns is out of bounds. Expected <= " + to_string(n)
						+ ", but got " + to_string(ne) + " instead.");
	}
	
	
	unsigned int sm = me - ms; //selection rows
	unsigned int sn = ne - ns; //selection cols
	
	dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	
	unsigned int blocksX = (unsigned int) ceil(sn / (double) BLOCK_SIZE_2D);
	unsigned int blocksY = (unsigned int) ceil(sm / (double) BLOCK_SIZE_2D);
	
	dim3 grid(blocksX, blocksY);
	
	kernelMatrixSelect<<<grid, block>>>(n, ms, ns, sm, sn, src, dest);

}

void copyVector(unsigned int length, float* src, float* dest) {
	
	unsigned int block = BLOCK_SIZE_1D;
	
	unsigned int grid = (unsigned int) ceil(length / (double) BLOCK_SIZE_1D);
	
	kernelCopyVector<<<grid, block>>>(length, src, dest);
}

void copyMatrix(unsigned int m, unsigned int n, float* src, float* dest) {
	checkCuda(cudaMemcpy(dest, src, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
}

void matrixInsertCol(unsigned int m, unsigned int n, unsigned int col, float val, float* src,
		float* dest) {
	
	dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	
	unsigned int blocksX = (unsigned int) ceil((n + 1) / (double) BLOCK_SIZE_2D);
	unsigned int blocksY = (unsigned int) ceil(m / (double) BLOCK_SIZE_2D);
	
	dim3 grid(blocksX, blocksY);
	
	kernelMatrixInsertCol<<<grid, block>>>(m, n + 1, col, val, src, dest);
}

void matrixInsertColVector(unsigned int m, unsigned int n, unsigned int col, float* vec, float* src,
		float* dest) {
	
	dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	
	unsigned int blocksX = (unsigned int) ceil((n + 1) / (double) BLOCK_SIZE_2D);
	unsigned int blocksY = (unsigned int) ceil(m / (double) BLOCK_SIZE_2D);
	
	dim3 grid(blocksX, blocksY);
	
	kernelMatrixInsertColVector<<<grid, block>>>(m, n + 1, col, vec, src, dest);
}

void matrixTranspose(unsigned int m, unsigned int n, float* src, float* dest) {
	
	dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	
	unsigned int blocksX = (unsigned int) ceil(n / (double) BLOCK_SIZE_2D);
	unsigned int blocksY = (unsigned int) ceil(m / (double) BLOCK_SIZE_2D);
	
	dim3 grid(blocksX, blocksY);
	
	kernelMatrixTranspose<<<grid, block>>>(m, n, src, dest);
}

