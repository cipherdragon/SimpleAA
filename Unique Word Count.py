# Importing needed libraries
import string
# import matplotlib



# Defining class for nodes to use in Binary Search Tree
class Node:
    def __init__(self, word, left=None, right=None):
        self.word = word
        self.left = left
        self.right = right
        self.freq = 1


# Defining class for Binary Search Tree
class BinarySearchTree:

    # Constructor of the class
    def __init__(self):
        self.root = None  # Initializing root node
        self.d = {}  # Defining dictionary to collect words and frequencies

    # Defining function for checking if Binary Search Tree is empty
    def empty(self):
        return self.root is None

    # Defining function for searching specific word
    def search(self, word):
        # Traversing Binary Search Tree and checking if word exists
        current = self.root
        while current is not None:
            # If word is found
            if word == current.word:
                # Returning word itself and its frequency
                return current.word, current.freq

            # Going left
            elif word < current.word:
                # Getting left child
                current = current.left

            # Going right
            else:
                # Getting right child
                current = current.right

        # If not found
        return False, False

    # Defining function for printing Binary Search Tree
    # Checking if root node is not empty
    def travers(self):
        if self.root is not None:
            # Adding node to the dictionary
            self.d[self.root.word] = self.root.freq
            # Traversing rest of the Binary Search Tree
            self._travers(self.root)

    # Traversing rest of the Binary Search Tree
    def _travers(self, current_node):
        # Checking if current node is not empty
        if current_node is not None:
            # Traversing left
            self._travers(current_node.left)

            # Adding current node to the dictionary
            self.d[current_node.word] = current_node.freq

            # Traversing right
            self._travers(current_node.right)

    # Defining function for adding new word
    # or increasing frequency counter if it exists already
    def add(self, word):
        if self.empty():
            self.root = Node(word)
            return

        # Keeping track of parent to use when adding
        parent = None

        # Traversing Binary Tree and checking if word exists
        current = self.root
        while current is not None:
            # Going left
            if word < current.word:
                # Before going to child, saving parent
                parent = current
                # Getting left child
                current = current.left

            # Going right
            elif word > current.word:
                # Before going to child, saving parent
                parent = current
                # Getting right child
                current = current.right

            # Increasing frequency if word exists already
            else:
                current.freq += 1
                return

        # Adding new word
        # New word will be a left child
        if word < parent.word:
            parent.left = Node(word)
        # New word will be a right child
        else:
            parent.right = Node(word)


# Defining main function for calculations
def main(sentence):
    """
    Counting words in the given sentence
    """

    # Defining counters
    counter_words = 0         # for words
    counter_unique_words = 0  # for unique words
    similar_words = 0         # for similar words

    # Initializing instance of the Binary Search Tree class
    bst = BinarySearchTree()

    # Preprocessing the sentence
    sentence = sentence.strip()
    sentence = sentence.translate(sentence.maketrans('', '', string.punctuation))
    sentence = sentence.lower()
    words = sentence.split()

    # Counting unique words and their frequencies
    for word in words:
        counter_words += 1
        bst.add(word)

    # Traversing Binary Search Tree and getting resulted dictionary
    # with unique words and their frequencies
    bst.travers()
    unique_words = bst.d

    # Updating counter for unique words
    counter_unique_words += len(unique_words)

    # Counting similar words
    for freq in unique_words.values():
        if freq > 1:
            similar_words += 1

    # Printing final results
    print('Total words = {}'.format(counter_words))
    print('Total unique words = {}'.format(counter_unique_words))
    print('Total similar words = {}'.format(similar_words))

    # Returning resulted dictionary, total words count, and similar words count
    return unique_words, counter_words, similar_words, counter_words


# Example usage:
sentence = input("Enter your sentence: ")
unique_words, total_words, similar_words, total_count = main(sentence)
print("Unique words and their frequencies:")
for word, freq in unique_words.items():
    print(f"{word}: {freq}")
print(f"Total words in the sentence: {total_words}")
print(f"Total similar words in the sentence: {similar_words}")
print(f"Total words in the sentence: {total_count}")