"""Allow running s3_tools as a module: python -m s3_tools"""

from .cli import main

if __name__ == '__main__':
    main()
