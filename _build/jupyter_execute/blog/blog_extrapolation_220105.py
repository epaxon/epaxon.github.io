#!/usr/bin/env python
# coding: utf-8

# # Interpolation and Extrapolation in Neural networks
# 
# 
# Watching machine learning street talk youtube channel, where they discuss with Yann Lecun whether neural networks only interpolate or that they can extrapolate.
# 
# https://www.youtube.com/watch?v=86ib0sfdFtw&ab_channel=MachineLearningStreetTalk
# 
# Some frustrating thoughts from one the "god-father" of deep learning. Just so much cringe...
# 
# There is also a paper he references: 
# https://arxiv.org/pdf/2110.09485.pdf
# 
# "Learning High Dimension Always Amounts to Extrapolation"
# 
# 
# They go into a discussion about even the definition of extrapolation and interpolation, and make some mistakes when thinking about high-dimensional space. 
# 
# There is a semantic discussion about what extrapolation even means. Interpolation is when a novel data-point exists inside the convex hull of data-points in the training set. Thus, if a point is within the convex hull of a data-point, then one can use a linear combination of the training points to explain the novel data-point, and this is considered (linear) interpolation.
# 
# But they argue that the likelihood of a data-point being inside the convex hull of training data decreases exponentially (or super-exponentially...) with the dimensionality of the data. Their argument amounts to considerations of the volumes of higher-dimensional object. This on its face is true, since the volume of high-dimensional space grows exponentially, convex hulls inside of this space take up proportionally less of the volume, making it exponentially less likely that a novel point will be inside of the convex hull.
# 
# They are building on a paper that argues that deep networks are not actually defining lower-dimensional latent spaces as a way to model the training data. Rather the data is split into independent volumes that are essentially categorical in nature, and the interpolation that we see is just an artifact of a large number of categories. 
# 
# However, this already doesn't jive well with the realities of a basis set. They are making a mistake that it only matters whats inside the hull, and that outside the hull is completely extraneous. But a basis set is not like this. 
# 
# A basis set and a latent space are essentially the same. Lets just consider images. We are always using a basis set to represent the image. If we just represented each pixel, then our basis functions are one-hot vectors the size of the image, with each vector representing one of the pixels. With $N$ pixel values we can represent any $N$-dimensional image. In essence, any image is a linear combination of the basis functions. 
# 
# We all know that through PCA or sparse coding we can come up with a more "efficient" basis set for images, where we reduce the total number of basis functions, and thus can represent a $N$-pixel image with fewer than $N$ basis functions. But this is because we are only considering "natural" images, which are compressible. There is of course some lost information when compressing images into a reduced basis set.
# 
# Regardless, it is actually quite trivial to use basis functions as a representation for images. The argument that a point is unlikely to fall into the convex-hull of the basis set is just wrong. Any image $N$-dimensional can be represented by a linear-combination of $N$ orthogonal basis functions. If I just enumerated a set of $N$ basis functions for each pixel, then every image would fall inside the convex hull, and every image would just be a linear-combination of these basis functions.

# In[ ]:




