# Just needs a C compiler. libEGL gets loaded when the program runs.
CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra

blackhole: render.c
	$(CC) $(CFLAGS) -pthread -o $@ $< -ldl -lm

clean:
	rm -f blackhole

.PHONY: clean
