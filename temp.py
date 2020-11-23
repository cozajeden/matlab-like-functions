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
            Chunk x dimension.
            default: 1
        
        y (int, optional):
            Chunk y dimension.
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

    while n <= N**2:
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
            ms = zeros((N,N), dtype=dt)
            sequence = [1 if i%4 in {0,3} else 0 for i in range(N)]
            N2 = N*N

            for x, i in enumerate(sequence):
                for y, j in enumerate(sequence):
                    n = x*N + y
                    if i == j:
                        ms[x, y] = n + 1
                    else:
                        ms[x, y] = N2 - n + 1
                    
        else:
            ms = zeros((N,N), dtype=dt)
            a = N//4
            b = a + 1
            c = int(N * 0.5)
            d = c * c
            e = a - 1
            f = N - 1

            chunk = __odd_magic(c, dt)
            ms[:c,:c] = chunk
            ms[c:,c:] = chunk + d
            ms[:c,c:] = chunk + d * 2
            ms[c:,:c] = chunk + d * 3

            for j in range(a):
                ms[a,j+1], ms[a+c, j+1] = ms[a+c, j+1], ms[a,j+1]
                for i in range(a):
                    ms[i,j],    ms[i+c,j]   = ms[i+c,j],    ms[i,j]
                    ms[i+b,j],  ms[i+b+c,j] = ms[i+b+c,j],  ms[i+b,j]
            for j in range(e):
                for i in range(c):
                    ms[i, f-j], ms[i+c,f-j] = ms[i+c,f-j], ms[i, f-j]

    else:
        raise ValueError(f'N have to be > 2, not({N})')
    return ms
    
def hufftree(HL, HK):
  N = len(HL) # number of symbols
  Htree = zeros((N*2, 3), uint8)
  next = 1
  for n in range(N):
    if HL[n] > 0:
      # place this symbol correct in Htree
      pos = 0
      for k in range(HL[n]):
        if Htree[pos, 0] == 0 and Htree[pos, 1] == 0:
          # it's a branching point but yet not activated
          Htree[pos, 1] = next 
          Htree[pos, 2] = next + 1
          next += 2
        if HK[n, k]:
          pos = Htree[pos, 2] # goto right branch
        else:
          pos = Htree[pos, 1] # goto left branch 
      Htree[pos, 0] = 1 # now the position is a leaf
      Htree[pos, 1] = n + 1 # and this is the symbol number it represent
    
  return Htree

def hufflen(S):
  HL = zeros_like(S)
  Sc = copy(S.flatten())
  Ip = argwhere(Sc > 0) # index of positive elements
  Sp = Sc[Sc > 0] # the positive elements of S
  N = len(Sp) # number of elements in Sp vector
  Ip = reshape(Ip, (1, N))
  HLp = zeros_like(Sp)
  C = append(Sp, zeros((N-1, 1), uint8)) # count or weights for each "tree"
  Top = array(range(N), uint8) # the "tree" every symbol belongs to
  So = sort(-Sp)
  Si = argsort(-Sp) # Si is indexes for descending symbols
  last = N - 1 # Number of "trees" now
  next = N # next free element in C 
  while last > 0:
    # the two smallest "trees" are put together
    C[next] = C[Si[last]] + C[Si[last - 1]]
    I = argwhere(Top == Si[last])
    HLp[I] += 1 # one extra bit added to elements in "tree"
    Top[I] = next
    I = argwhere(Top == Si[last - 1])
    HLp[I] += 1
    Top[I] = next
    last -= 1
    Si[last] = next
    next += 1
    count = last -1
    while count > -1 and C[Si[count + 1]] >= C[Si[count]]:
      temp = Si[count]
      Si[count] = Si[count + 1]
      Si[count + 1] = temp
      count -= 1
  HL[Ip] = HLp
  return HL

def huffcode(HL, Display=False):
  """ Based on the codeword lengths this function find the Huffman codewords

  HK = huffcode(HL,Display);
  HK = huffcode(HL);
  ------------------------------------------------------------------
  Arguments:
    HL     length (bits) for the codeword for each symbol 
          This is usually found by the hufflen function
    HK     The Huffman codewords, a matrix of ones or zeros
          the code for each symbol is a row in the matrix
          Code for symbol S(i) is: HK(i,1:HL(i))
          ex: HK(i,1:L)=[0,1,1,0,1,0,0,0] and HL(i)=6 ==> 
              Codeword for symbol S(i) = '011010'
    Display==1  ==> Codewords are displayed on screen, Default=0
  ------------------------------------------------------------------
  ----------------------------------------------------------------------
  Copyright (c) 1999.  Karl Skretting.  All rights reserved.
  Hogskolen in Stavanger (Stavanger University), Signal Processing Group
  Mail:  karl.skretting@tn.his.no   Homepage:  http://www.ux.his.no/~karlsk/
  
  HISTORY:
  Ver. 1.0  25.08.98  KS: Function made as part of Signal Compression Project 98
  Ver. 1.1  25.12.98  English version of program
  Ver. 1.1p 26.10.20  Python version of program
  ----------------------------------------------------------------------"""

  N = len(HL)
  L = max(HL)
  HK = zeros((N, L))
  HLi = argsort(HL)
  HLs = sort(HL)
  Code = zeros((L))
  for n, hls in enumerate(HLs):
    if hls > 0:
      HK[HLi[n]] = Code
      k = HLs[n] -1;
      while k > -1:
        Code[k] += 1
        if Code[k] == 2:
          Code[k] = 0
          k -= 1
        else:
          break
  if Display is not False:
    for i, n in enumerate(zip(HK, HL)):
      print(f'Symbol {i:15d} gets code: {n[0][:n[1]]}')
  return HK


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
    y = magic(21)
    plt.subplot(323)
    plt.imshow(y)
    plt.title('IN')
    y = blockproc(y, foo, 3, 3)
    plt.subplot(324)
    plt.imshow(y)
    plt.title('OUT')
    y = arange(2).reshape((1, 2))
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