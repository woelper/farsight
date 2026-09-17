# Base training corpora

Public-domain books from Project Gutenberg (US public domain; authors long
deceased), license headers stripped (body kept verbatim otherwise).

## English (`base_en.txt`, ~133k tokens)

- *Alice's Adventures in Wonderland* — Lewis Carroll (d. 1898)
  https://www.gutenberg.org/files/11/11-0.txt
- *The Adventures of Sherlock Holmes* — Arthur Conan Doyle (d. 1930)
  https://www.gutenberg.org/files/1661/1661-0.txt

## German (`base_de.txt`, ~85k tokens)

- *Also sprach Zarathustra* — Friedrich Nietzsche (d. 1900)
  https://www.gutenberg.org/files/7205/7205-0.txt

## Sample (`sample_en_de.txt`, 353 tokens)

Tiny hand-written EN+DE mix for fast eval regression tests — NOT training
data for the daemon (too small, deliberately repetitive).
