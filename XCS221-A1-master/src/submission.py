#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 1: binary classification
############################################################

############################################################
# Problem 1a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    pass
    # ### START CODE HERE ###
    feature_vector: FeatureVector = {}
    
    # Divide la cadena en palabras utilizando el espacio en blanco como delimitador
    words = x.split()
    
    # Cuenta la frecuencia de cada palabra y actualiza el vector de características
    for word in words:
        feature_vector[word] = feature_vector.get(word, 0) + 1
    return feature_vector
    # ### END CODE HERE ###


############################################################
# Problem 1b: stochastic gradient descent

T = TypeVar("T")


def learnPredictor(
    trainExamples: List[Tuple[T, int]],
    validationExamples: List[Tuple[T, int]],
    featureExtractor: Callable[[T], FeatureVector],
    numEpochs: int,
    eta: float,
) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    """
    weights = {}  # feature => weight
    # ### START CODE HERE ###
    for epoch in range(numEpochs):
        for x, y in trainExamples:
            features = featureExtractor(x)
            score = dotProduct(features, weights)
            if y * score < 1:
                increment(weights, eta * y, features)

        trainError = evaluatePredictor(trainExamples, lambda x: 1 if dotProduct(featureExtractor(x), weights) >= 0 else -1)
        print(f"Epoch {epoch+1}: Training Error = {trainError}")

        validationError = evaluatePredictor(validationExamples, lambda x: 1 if dotProduct(featureExtractor(x), weights) >= 0 else -1)
        print(f"Validation Error = {validationError}")
    # ### END CODE HERE ###
    return weights



############################################################
# Problem 1c: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = None
        y = None
        # ### START CODE HERE ###
        phi = {}
        # Generate phi(x)
        for key in weights.keys():
            # Randomly assign values to features
            phi[key] = random.randint(-10, 10)

        # Calculate the score
        score = dotProduct(phi, weights)

        # Determine y
        if score == 0:
            y = 1
        else:
            y = 1 if score > 0 else -1
        # ### END CODE HERE ###
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 1d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x):
        pass
        # ### START CODE HERE ###
        x = x.replace(" ", "").replace("\t", "")
        # Create an empty feature vector
        features = {}

        # Extract all n-grams of length n from the input string
        for i in range(len(x) - n + 1):
            n_gram = x[i:i + n]
            # Increment the count of the n-gram in the feature vector
            features[n_gram] = features.get(n_gram, 0) + 1

        return features
        # ### END CODE HERE ###

    return extract


############################################################
# Problem 1e:
#
# Helper function to test 1e.
#
# To run this function, run the command from termial with `n` replaced
#
# $ python -c "from submission import *; testValuesOfN(n)"
#


def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be submitted.
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
            "Official: train error = %s, validation error = %s"
            % (trainError, validationError)
        )
    )


############################################################
# Problem 2b: K-means
############################################################


def kmeans(examples: List[Dict[str, float]], K: int, maxEpochs: int) -> Tuple[List[Dict[str, float]], List[int], float]:
    """
    Perform k-means clustering.

    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """
    # Initialize cluster centroids to random examples
    centers = random.sample(examples, K)
    assignments = [-1] * len(examples)  # Initialize assignments to -1 indicating unassigned

    # Precompute distances between examples and centroids
    distances = [[sum((examples[i].get(key, 0.0) - center.get(key, 0.0)) ** 2 for key in set(examples[i]) | set(center)) for center in centers] for i in range(len(examples))]

    for epoch in range(maxEpochs):
        # Assign examples to the nearest cluster using precomputed distances
        new_assignments = [min(range(K), key=lambda j: distances[i][j]) for i in range(len(examples))]

        # Update cluster centroids
        new_centers = []
        for j in range(K):
            cluster_indices = [i for i, assignment in enumerate(new_assignments) if assignment == j]
            if cluster_indices:
                cluster_examples = [examples[i] for i in cluster_indices]
                cluster_center = {}
                for key in set().union(*cluster_examples):
                    cluster_center[key] = sum(example.get(key, 0.0) for example in cluster_examples) / len(cluster_examples)
                new_centers.append(cluster_center)

        # Check for convergence
        if new_assignments == assignments and new_centers == centers:
            break

        assignments = new_assignments
        centers = new_centers

    # Calculate final reconstruction loss
    total_loss = sum(sum((examples[i].get(key, 0.0) - centers[assignments[i]].get(key, 0.0)) ** 2 for key in set(examples[i]) | set(centers[assignments[i]])) for i in range(len(examples)))

    return centers, assignments, total_loss

