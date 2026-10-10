# Just needs a C compiler. libEGL gets loaded when the program runs.
CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra

geodesic: main.c
	$(CC) $(CFLAGS) -o $@ $< -ldl -lm

clean:
	rm -f geodesic

.PHONY: clean
