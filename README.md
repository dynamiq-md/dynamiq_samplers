[![Build Status](https://travis-ci.org/dynamiq-md/dynamiq_samplers.svg?branch=master)](https://travis-ci.org/dynamiq-md/dynamiq_samplers)
[![Coverage Status](https://coveralls.io/repos/github/dynamiq-md/dynamiq_samplers/badge.svg?branch=master)](https://coveralls.io/github/dynamiq-md/dynamiq_samplers?branch=master)

# dynamiq_samplers

MC samplers for dynamiq-md projects

To start, this will contain initial condition samplers. At the beginning, this is just sampling from Gaussians (which ends up being a very useful way to do sampling for SC-IVR calculations).

This will eventually be united into a common framework with the samplers developed in OpenPathSampling. After all, initial condition sampling of trajectories is just a subset of path sampling.

These particular samplers are specifically designed to support Monte Carlo integration schemes, as used in SC-IVR calculations.
