import torch
import cupy

kernel_gather_operation = '''
extern "C" __global__ void kernel_gather_operation(int b, int c, int n, int m,
                                     const float *__restrict__ points,
                                     const int *__restrict__ idx,
                                     float *__restrict__ out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}
'''

kernel_gather_grad = '''
extern "C" __global__ void kernel_gather_grad(int b, int c, int n, int m,
                                          const float *__restrict__ grad_out,
                                          const int *__restrict__ idx,
                                          float *__restrict__ grad_points) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        atomicAdd(grad_points + (i * c + l) * n + a,
                  grad_out[(i * c + l) * m + j]);
      }
    }
  }
}
'''


def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]
    return strKernel
# end

@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)

class FunctionGatherOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        B, C, N = features.size()
        _, npoint = idx.size()
        out = torch.cuda.FloatTensor(B, C, npoint).fill_(0)


        ctx.for_backwards = (idx, C, N)

        if features.is_cuda == True:
            cupy_launch('kernel_gather_operation', cupy_kernel('kernel_gather_operation', {
                'b': B,
                'c': C,
                'n': N,
                'm': npoint,
                'points': features,
                'idx': idx,
                'out': out
            }))(
                grid=tuple([ B, C, 1 ]),
                block=tuple([ 512, 1, 1 ]),
                args=[ B, C, N, npoint, features.data_ptr(), idx.data_ptr(), out.data_ptr() ]
            )

        return out

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        B = grad_out.size(0)
        npoint = idx.size(1)
        grad_features = torch.cuda.FloatTensor(grad_out.size(0), grad_out.size(1), N).fill_(0)

        if grad_out.is_cuda == True:
            cupy_launch('kernel_gather_grad', cupy_kernel('kernel_gather_grad', {
                'b': B,
                'c': C,
                'n': N,
                'm': npoint,
                'grad_out': grad_out,
                'idx': idx,
                'grad_points': grad_features
            }))(
                grid=tuple([ B, C, 1 ]),
                block=tuple([ 512, 1, 1 ]),
                args=[ B, C, N, npoint, features.data_ptr(), idx.data_ptr(), out.data_ptr() ]
            )

        # grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


class GatherOperation(torch.nn.Module):
    def __init__(self):
        super(GatherOperation, self).__init__()

    def forward(self, features, idx):
        return FunctionGatherOperation.apply(features, idx)




