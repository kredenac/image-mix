#%%

import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy import ndimage
import math
import matplotlib.pyplot as plt

def makeGaussianFilter(numRows, numCols, sigma, highPass=True):
   centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
   centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)
 
   def gaussian(i,j):
      coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
      return 1 - coefficient if highPass else coefficient
 
   return numpy.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])
 
def filterDFT(imageMatrix, filterMatrix):
   shiftedDFT = fftshift(fft2(imageMatrix))
   
   filteredDFT = shiftedDFT * filterMatrix

   plt.imshow(filteredDFT.real, cmap='gray')
   plt.show()

   res = ifft2(ifftshift(filteredDFT))

   return res
 
def lowPass(imageMatrix, sigma):
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=False))
 
def highPass(imageMatrix, sigma):
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=True))
 
def hybridImage(highFreqImg, lowFreqImg, sigmaHigh, sigmaLow):
   highPassed = highPass(highFreqImg, sigmaHigh)
   lowPassed = lowPass(lowFreqImg, sigmaLow)
 
   return highPassed/2 + lowPassed/4

#%%
marilyn = ndimage.imread("marilyn.png", flatten=True)
einstein = ndimage.imread("albert.png", flatten=True)

hybrid = hybridImage(einstein, marilyn, 20, 20)
misc.imsave("mix.png", numpy.real(hybrid))

#%%
def showImg(img):
   plt.imshow(img, cmap='gray')
   plt.show()

imageMatrix = marilyn
showImg(marilyn)

n,m = imageMatrix.shape
filterMatrix = makeGaussianFilter(n, m, 200, highPass=True)


shiftedDFT = fftshift(fft2(imageMatrix))
plt.title("shiftedDFT.real")
showImg(shiftedDFT.real)
# plt.title("shiftedDFT.imag")
# showImg(shiftedDFT.imag)

DFT = fft2(imageMatrix) 
plt.title("DFT.real")
showImg(DFT.real)
# plt.title("DFT.imag")
# showImg(DFT.imag)

filteredDFT = shiftedDFT * filterMatrix
plt.title("filteredDFT.real")
showImg(filteredDFT.real)
plt.title("filteredDFT.imag")
showImg(filteredDFT.imag)

filteredUnshifted = DFT * filterMatrix
plt.title("filteredUnshifted.real")
showImg(filteredUnshifted.real)
plt.title("filteredUnshifted.imag")
showImg(filteredUnshifted.imag)

res2 = ifft2(filteredUnshifted)
plt.title("res2.real")
showImg(res2.real)
plt.title("res2.imag")
showImg(res2.imag)

res = ifft2(ifftshift(filteredDFT))
plt.title("res.real")
showImg(res.real)
plt.title("res.imag")
showImg(res.imag)


#%%
test = [0,1,2]
print(test)
out = fftshift(test)
print(out)
