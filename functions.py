from types import FunctionType
from numpy import *
from numpy import asarray
from PIL.Image import open
from PIL.ImageOps import grayscale

def blockproc(a: ndarray or str or list or tuple, fun=lambda x: x, x=1, y=1, gs=False) -> array:
    """
    #### Will return `array` processed in chunks with shape (`x`, `y`).
    
    ### Parameters:

        a (numpy.ndarray, str, list, tuple): 
            If numpy.ndarray, list or tuple dimensions have to be 1D - 3D.
            If str - path to an image file.
            
        fun (FunctionType, optional):
            Function to use on each block.
            A Function has to return numpy.ndarray with a dimensions 1D - 3D and any size.
            default: lambda x: x
        
        x (int, optional):
            Chunk Axis-0 dimension.
            default: 1
        
        y (int, optional):
            Chunk Axis-1 dimension.
            default: 1

        gs (bool, optional):
            If True and arg 1 is a path, the image will be in grayscale.
            default: False
    
    ### Returns:
    
        numpy.ndarray:
            Array after applying a function 'fun' to each chunk of a size (x, y).
    """
    # Test the parameters against a types.
    if isinstance(a, (list, tuple)):
        a = array(a)
    elif isinstance(a, str):
        if gs:  a = asarray(grayscale(open(a)))
        else:   a = asarray(open(a))
    elif not isinstance(a, ndarray):
        raise   TypeError(f'arg 1: should be a {type(ndarray)} or a {type(str)}, not the {type(a)}.')
    # Make sure the array will be 2D at least.
    if len(a.shape) == 1:
        a = a.reshape((a.shape[0], 1))

    # Test the parameters against a shape
    assert x > 0 and y > 0 and not a.shape[0]%x and not a.shape[1]%y, \
        f"A block with shape ({x}, {y}) can't be used for an array with shape {str(a.shape)}."

    # Get the first chunk.
    lenx   = int(a.shape[0]/x)
    leny   = int(a.shape[1]/y)
    chunk  = fun(a[ 0:x, 0:+y ])

    # Test the first chunk against a type.
    try:
        b = array(chunk)
    except:
        raise TypeError(f"A 'fun' output have to be compatible with a numpy.ndarray.")

    # Get the new chunk shape.
    shapec = chunk.shape
    if len(shapec) == 1:    shapec = (shapec[0], 1)
    elif len(shapec) == 0:  shapec = (1, 1)
    newx, newy = shapec[:2]

    # Create an empty array with the output dimensions and the chunk data type.
    b = empty((newx*lenx, newy*leny, *shapec[2:]), dtype=chunk.dtype)

    # Put the first chunk into the output array.
    b[ 0:newx, 0:newy] = chunk

    # Calculate the rest of the chunks.
    for i in range(leny):
        for j in range(lenx):
            if i == 0 and j == 0:
                continue # Skip the first chunk. It is already calculated.
            else:
                b[ j*newx:(j + 1)*newx, i*newy:(i+ 1)*newy ] = fun(a[ j*x:(j + 1)*x, i*y:(i+1)*y ])
    return b

def __odd_magic(N, dt=int):
    n = 1
    i, j = 0, N//2
    magic_square = zeros((N,N), dtype=dt)

    while n <= N*N:
        magic_square[i, j] = n
        n += 1
        newi, newj = (i-1) % N, (j+1)% N
        if magic_square[newi, newj]:
            i += 1
        else:
            i, j = newi, newj
    
    return magic_square

def magic(N=3, dt=int) -> array:
    """Create an N x N magic square.""" 
    ms = None

    if N > 2:
        if N%2 == 1:
            ms = __odd_magic(N, dt)

        elif N%4 == 0:
            N2 = N*N
            con = lambda x: x%4 in {0,3}
            mask = array([[True if con(x)^con(y) else False for y in range(N)] for x in range(N)]).reshape((N, N))
            ms = arange(1,N2 + 1, dtype=dt).reshape((N, N))
            ms[mask] = N2 - ms[mask]
                    
        else:
            ms = zeros((N,N), dtype=dt)
            a = N//4
            b = a + 1
            c = N//2
            d = c * c
            e = a - 1

            chunk = __odd_magic(c, dt)
            ms[:c,:c] = chunk
            ms[c:,c:] = chunk + d
            ms[:c,c:] = chunk + d * 2
            ms[c:,:c] = chunk + d * 3
            s = ms.copy()
            
            ms[c:a+c,:a], ms[:a,:a]        = s[:a,:a],         s[c:a+c,:a]
            ms[b:a+b,:a], ms[b+c:a+b+c,:a] = s[b+c:a+b+c, :a], s[b:a+b,:a]
            ms[a,1:b],    ms[a+c,1:b]      = s[a+c,1:b],       s[a,1:b]
            ms[:c,N-e:N], ms[c:, N-e:N]    = s[c:, N-e:N],     s[:c,N-e:N]

    else:
        raise ValueError(f'N have to be > 2, not({N})')
    return ms

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    bar = lambda x: x.dot([[1/3]]*3)
    def foo(x): return sum(x)/3
    
    plt.figure(0)

    y = magic(99)
    plt.subplot(321)
    plt.imshow(y)
    plt.title('IN')
    y = blockproc(y, bar, 3, 3)
    plt.subplot(322)
    plt.imshow(y)
    plt.title('OUT')
    y = magic(60)
    plt.subplot(323)
    plt.imshow(y)
    plt.title('IN')
    y = blockproc(y, foo, 3, 3)
    plt.subplot(324)
    plt.imshow(y)
    plt.title('OUT')
    y = arange(6).reshape((2, 3))
    plt.subplot(325)
    plt.imshow(y)
    plt.title('IN')
    y = blockproc(y, lambda x: magic(14))
    plt.subplot(326)
    plt.imshow(y)
    plt.title('OUT')

    plt.figure(1)

    def bar(a):
        x, y = a.shape[:2]
        b = a.copy()
        x = x//2
        y = y//2
        b[:x, :y], b[x:, y:] = b[x:, y:], b[:x, :y]
        b[x:, :y], b[:x, y:] = b[:x, y:], b[x:, :y]
        return b
    def foo(x):
        z = zeros_like(x)
        for i in range(3):
            z[:,:,i] = x[:,:,i] * magic(z.shape[0])/(z.shape[0] * z.shape[0])
        return z
    def fun(x):
        x = sum(x)/(x.shape[0]*x.shape[1])
        return x

    y = blockproc('lena512color.tiff')
    plt.subplot(321)
    plt.imshow(y)
    plt.title('IN')
    x = blockproc(y, bar, 8, 8)
    plt.subplot(322)
    plt.imshow(x)
    plt.title('OUT')
    plt.subplot(323)
    plt.imshow(y)
    plt.title('IN')
    x = blockproc(y, foo, 128, 128)
    plt.subplot(324)
    plt.imshow(x)
    plt.title('OUT')
    plt.subplot(325)
    plt.imshow(y)
    plt.title('IN')
    x = blockproc(y, fun, 16,16)
    plt.subplot(326)
    plt.imshow(x)
    plt.title('OUT')
    plt.show()