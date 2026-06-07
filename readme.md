# Trellis: An unconventional application of the shortest path

This program uses a trellis graph in combination with an
[n-gram database](https://norvig.com/mayzner.html) and Dijkstra's algorithm
to find the most likely letter combination that corresponds to a number
sequence associated with a keypad. It was implemented as part of the MAA600
Graph Theory, Networks and Applications course at Mälardalen University.

## Usage
```sh
# Run this for a help message describing how to call the program.
uv run trellis.py -h

# Turns the word "hello" into a number sequence, then runs the algorithm on the
# number sequence and prints the most probable letter sequence it corresponds
# to.
uv run trellis.py -t hello
```
