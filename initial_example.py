import xlns as xl
import numpy as np

def main():
    x = xl.xlnsnp([2.0, 3.0])
    y = xl.xlnsnp([2.0, 4.0])
    
    print("x:", x, "x integer-form:", x.nd)
    print("y:", y, "y integer-form:", y.nd)
    
    print("Sum:", x+y, "Sum integer-form:", x.nd+y.nd)
    print("Difference:", x-y, "Difference integer-form:", x.nd-y.nd)
    print("Product:", x*y, "Product integer-form:", x.nd*y.nd)
    print("Quotient:", x/y, "Quotient integer-form:", x.nd*y.nd)

if __name__ == "__main__":
    main()