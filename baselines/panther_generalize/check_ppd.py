import spu.utils.distributed as ppd
import argparse
import json

def main():
    print("Checking ppd stats capabilities...")
    # Just print dir of ppd to see if we can find something like get_stats
    print(dir(ppd))

if __name__ == "__main__":
    main()

