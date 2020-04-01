---
layout: single
author_profile: true
title: Good practices with numpy random number generators
excerpt_separator: <!--more-->
---

Unless you are working on a problem where you can afford a true Random Number Generator (RNG), implementing something random means relying on a pseudo Random Number Generator. I want to share here what I have learnt about how to properly use a pseudo RNG and especially the ones available in [numpy](https://numpy.org/). <!--more--> I assume a certain knowledge of numpy and that numpy 1.17 or greater is used. The reason for this is that great new features were introduced in the [random](https://numpy.org/doc/1.18/reference/random/index.html) module in version 1.17. As `numpy` is usually imported as `np`, I will sometimes use `np` instead of `numpy`. Finally, as I will not talk about true RNGs, RNG will always mean pseudo RNG in the rest of this blog post.

<!---
A lot of computation in machine learning rely on randomness, including data generation, data preprocessing, cross-validation, optimization algorithms such as stochastic gradient descent, random initialization (for instance for neural networks). One also wants to know whether his results hold independently of this randomness. Specifically, the results will most likely be true on average or with great probability. To assess the performance of his algorithm or idea, one usually repeats the experiment several times.

Finally, we want to be able to reproduce our results, for the sake of science but also more simply for the sake of our jobs of debuggers.
-->

### The main messages
1. Avoid using the global numpy RNG and hence avoid using [`np.random.seed`](https://numpy.org/doc/1.18/reference/random/generated/numpy.random.seed.html) or `np.random.*` functions to generate random values such as `np.random.rand`.
2. Create a new RNG and pass it around using the [`np.random.default_rng`](https://numpy.org/doc/1.18/reference/random/generator.html?highlight=numpy%20random%20default_rng#numpy.random.default_rng) function.
3. Be careful with parallel computations and rely on [numpy strategies for reproducible parallel number generation](https://numpy.org/doc/1.18/reference/random/parallel.html).

Note that before numpy 1.17 the way to create a new RNG was to use [`np.random.RandomState`](https://numpy.org/doc/1.18/reference/random/legacy.html#numpy.random.RandomState) which is based on the Mersenne Twister 19937 algorithm. This is also how the global numpy RNG is created. It is still possible to use this function in versions higher than 1.17 but it is now recommended to use `default_rng` which returns an instance of the statistically better PCG64 RNG.

## Random number generation with numpy
When you import `numpy` in your python script a RNG is created behind the scenes. This RNG is the one used when you generate a new random value using a function such as `np.random.random`. I will here refer to this RNG as the global numpy RNG.

It is a common practice to reset the seed of this global RNG at the beginning of a script using the `np.random.seed` function. Fixing the seed at the beginning ensures that the script is reproducible: the same values and results will be produced each time you run it. However, although sometimes convenient, using the global numpy RNG is considered a bad practice. A simple reason is that using global variables can lead to undesired side effects. For instance one might use `np.random` without knowing that the seed of the global RNG was reset somewhere else in the codebase. Quoting the [Numpy Enhancement Proposal (NEP) 19](https://numpy.org/neps/nep-0019-rng-policy.html) by Robert Kern about the numpy RNG policy:

> The implicit global RandomState behind the `np.random.*` convenience functions can cause problems, especially when threads or other forms of concurrency are involved. Global state is always problematic. We categorically recommend avoiding using the convenience functions when reproducibility is involved. [...] The preferred best practice for getting reproducible pseudorandom numbers is to instantiate a generator object with a seed and pass it around.

In short:
* Instead of using `np.random.seed`, which reseeds the already created global numpy RNG and then using `np.random` functions you should create a new RNG.
* You should create one RNG at the beginning of your script (with a seed if you want reproducibility) and use this RNG in the rest of your script.

To create a new RNG you can use the `np.random.default_rng` function as illustrated in the [introduction of the random module documentation](https://numpy.org/doc/1.18/reference/random/index.html#introduction):

```python
from numpy.random import default_rng
seed = 12345
rng = default_rng(seed)  # can be called without a seed
rng.random()
```

The reason for seeding your RNG only once is that you can loose on the randomness and the independence of the generated random numbers by reseeding the RNG multiple times. Furthermore obtaining a good seed can be time consuming. Once you have a good seed to instantiate your generator you might as well use it. With a good RNG such as the one of numpy you will be ensured good randomness (and independence) of the generated numbers. It might be more dangerous to use different seeds: how do you know that the streams of random numbers obtained with two different seeds are not correlated, or I should say less independent than the ones created from the same seed? That being said, [as explained by Robert Kern](https://github.com/numpy/numpy/issues/15322#issuecomment-573890207), with the RNGs and seeding strategies introduced in numpy 1.17, it could be considered safe enough to recreate new RNGs from the system entropy, e.g. using `default_rng(None)` multiple times. However as explained later be careful when running jobs in parallel and relying on `default_rng(None)`.


### Passing a numpy RNG to your own functions

As you write functions that you will use on their own as well as in a more complex script it is convenient to be able to pass a seed or your already created RNG. The function `np.random.default_rng` allows you to do this very easily. As written above, this function can be used to create a new RNG from your chosen seed, if you pass a seed to it, or from system entropy when passing `None` but you can also pass an already created RNG. In this case the returned RNG is the one that you passed.

```python
from numpy.random import default_rng

def stochastic_function(seed, high=10):
    rng = default_rng(seed)
    return rng.integers(high, size=5)
```
You can either pass a fixed seed or your already created RNG to this function. To be perfectly exact, the `default_rng` function returns the exact same RNG passed to it for certain kind of RNGs such at the ones created with `default_rng` itself. You can refer to the [`default_rng` documentation](https://numpy.org/doc/1.18/reference/random/generator.html#numpy.random.default_rng) for more details on the arguments that you can pass to this function. You can also pass a `SeedSequence` instance. More on this below. 

Before knowing about `default_rng`, and before numpy 1.17, I was using the scikit-learn function [`sklearn.utils.check_random_state`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html) which is of course heavily used in the scikit-learn codebase. While writing this post I discovered that this function is now available in [scipy](https://github.com/scipy/scipy/blob/master/scipy/_lib/_util.py#L173). A look at the docstring and/or the source code of this function will give you a good idea about what it does. The only difference with `default_rng` is that when `None` is passed to `check_random_state` then the function returns the already existing global numpy RNG. This can be convenient because if you fixed the seed before in your script using `np.random.seed`, the function returns the generator that was seeded at the beginning of the script. However, as explained above, this is not the recommended practice and you should be aware of the risks and the side effects.

**add note on new API, use default_rng, Generator, instead of np.random.RandomState**


## Parallel processing

You must be careful when using RNGs in conjunction with parallel processing. I usually use the [joblib](https://joblib.readthedocs.io/en/latest/index.htl) library to parallelize my code and I will therefore mainly talk about it. However most of the discussion is not specific to joblib.

Depending on the parallel processing library or backend that you use different behaviors can be observed. For instance if you rely on the global numpy RNG to generate random numbers in parallel, it can be the case that forked Python processes use the same random seed and thus produce the exact same results which is a waste of computational resources. I learnt this the hard way during my PhD. A very nice example showing the different behaviors that one can obtain with joblib is available [here](https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html).

As written in the introduction of this article, it is a good practice to fix the seed for reproducibility. If you fix the seed at the beginning of your main script and then pass the same RNG to each process to be run in parallel, most of the time this will not give you what you want as this RNG will be deep copied and thus the same results will be produced by each process. One of the solutions is to create as many RNGs as parallel processes and choose one different seed for each of these RNGs. The issue now is that you cannot choose the seeds as easily as you would think. When you choose two different seeds to instantiate two different RNGs how do you know that the numbers produced by these RNGs will appear as statistically independent? The design of independent RNGs for parallel processes has been an important research question. You can for instance refer to the paper [Random numbers for parallel computers: Requirements and methods, with emphasis on GPUs](https://www.sciencedirect.com/science/article/pii/S0378475416300829) by L'Ecuyer et al. (2017) for a good summary of the different methods on this topic.

From numpy 1.17, it is now very easy to instantiate independent RNGs. Depending on the RNG that you use, different strategies are easily available as documented in the [Parallel random number generation section](https://numpy.org/doc/1.18/reference/random/parallel.html) of the numpy documentation. One of the strategies is to use `SeedSequence` which is an algorithm that makes sure that a not so good user-provided seed results in a good initial state for the RNG. Additionally, it ensures that two close seeds will result in two very different initial states for the RNG that are independent of each other. You can refer to the documentation of [SeedSequence Spawning](https://numpy.org/doc/1.18/reference/random/parallel.html#seedsequence-spawning) for an example on how to generate independent RNGs from a user-provided seed. I here provide an example illustrating how to use this with the [joblib example](https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html#fixing-the-random-state-to-obtain-deterministic-results) mentioned above.


```python
from numpy.random import SeedSequence, default_rng
from joblib import Parallel, delayed

def stochastic_function(seed, high=10):
    rng = default_rng(seed)
    return rng.integers(high, size=5)

seed = 98765
ss = SeedSequence(seed)
# create 5 child SeedSequences, one for each process.
child_states = ss.spawn(5)

random_vector = Parallel(n_jobs=2)(delayed(
    stochastic_function)(random_state) for random_state in child_states)
print(random_vector)

random_vector = Parallel(n_jobs=2)(delayed(
    stochastic_function)(random_state) for random_state in child_states)
print(random_vector)
```

By using a fixed random state you always get the same results and by using `SeedSequence.spawn` you have an independent RNG for each of the processes. Note that I also used the convenient `default_rng` function in `stochastic_function`.

## Resources

### Numpy RNGs
* [The documentation of the numpy random module](https://numpy.org/doc/1.18/reference/random/index.html) is the best place to find information and where I found most of the information that I shared here.
* [The Numpy Enhancement Proposal (NEP) 19 on the Random Number Generator Policy](https://numpy.org/neps/nep-0019-rng-policy.html) which lead to the changes introduced in numpy 1.17
* A [recent numpy issue](https://github.com/numpy/numpy/issues/15322) about the `check_random_state` and RNG good practices, especially [this comment](https://github.com/numpy/numpy/issues/15322#issuecomment-573890207) by Robert Kern.
* [How do I set a random_state for an entire execution?](https://scikit-learn.org/stable/faq.html#how-do-i-set-a-random-state-for-an-entire-execution) from the scikit-learn FAQ.

### RNGs in general
* [Random numbers for parallel computers: Requirements and methods, with emphasis on GPUs](https://www.sciencedirect.com/science/article/pii/S0378475416300829) by L'Ecuyer et al. (2017)
* To know more about the default RNG used in numpy, named PCG, I recommend the [PCG paper](https://www.pcg-random.org/paper.html) which also contains lots of useful information about RNGs in general. The [pcg-random.org website](https://www.pcg-random.org) is also full of interesting information about RNGs.