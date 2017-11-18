// Catherine Galkina, group 624, year 2017

#include <math.h>
#include <stdio.h>
#include <mpi.h>

#define N1_DEFAULT 1000
#define N2_DEFAULT 1000

// П = [0,2]x[0,2]
const double A1 = 0;
const double A2 = 2;
const double B1 = 0;
const double B2 = 2;

// Step q=3/2
const double q = 1.5;

// Eps = 10^-4
const double eps = 0.0001;

int N1, N2;

double F(double x, double y)
{
    double xx = x*x, yy = y*y;
    return 2*(xx + yy)*(1 - 2*xx*yy)*exp(1 - 2*xx*yy);
}

double phi(double x, double y)
{
    double xx = x*x, yy = y*y;
    return exp(1 - 2*xx*yy);
}

double f_node(double t)
{
    return (pow(1 + t, q) - 1)/(pow(2.0, q) - 1);
}

double x_i(int i)
{
    double f = f_node((double)i/N1);
    return A2*f + A1*(1 - f);
}

double y_j(int j)
{
    double f = f_node((double)j/N2);
    return B2*f + B1*(1 - f);
}

void distribute_points(int n, int rank, int size, int &start, int &end)
{
    // n1*procs1 + n2*procs2 = n
    int procs1 = n % size;
    int n1 = n/size + 1;
    int procs2 = n - procs1;
    int n2 = n/size;
    if (rank < procs1) {
        start = rank*n1;
        end = start + n1 - 1;
    } else {
        start = procs1*n1 + (rank - procs1)*n2;
        end = start + n2 - 1;
    }
}

class Node
{
    // Rank
    int rank;
    // Column
    int i;
    // Row
    int j;

    // Ranks of neighbours
    int left;
    int right;
    int up;
    int down;

    // Points distribution
    int x1;
    int x2;
    int y1;
    int y2;
    double *xs;
    double *ys;

public:
    Node(int r, int rows, int cols)
    {
        rank = r;
        j = r/cols;
        i = r - j*cols;

        int left_i = (i-1 >= 0) ? i-1 : cols - 1;
        int right_i = (i+1)%cols;
        int up_j = (j-1 >= 0) ? j-1 : rows - 1;
        int down_j = (j+1)%rows;
        left = (rank%cols) ? rank - 1 : -1;
        right = ((rank+1)%cols) ? rank + 1 : -1;
        up = (rank-cols >= 0) ? rank - cols : -1;
        down = (rank+cols < rows*cols) ? rank + cols : -1;

        distribute_points(N1, i, cols, x1, x2);
        distribute_points(N2, j, rows, y1, y2);
        xs = new double [x2 - x1 + 1];
        ys = new double [y2 - y1 + 1];

        for (int xi = x1; xi <= x2; xi++)
            xs[xi-x1] = x_i(xi);
        for (int yj = y1; yj <= y2; yj++)
            ys[yj-y1] = y_j(yj);
    }
    void Print()
    {
        const char* format
            = "%d i %d\n"
              "%d j %d\n"
              "%d left %d\n"
              "%d right %d\n"
              "%d up %d\n"
              "%d down %d\n"
              "%d x1 %d x2 %d\n"
              "%d y1 %d y2 %d\n";
        printf(format, rank, i, rank, j, rank, left,
                rank, right, rank, up, rank, down,
                rank, x1, x2,
                rank, y1, y2);
    }
};

void distribute_procs(int procs, int &rows, int &cols)
{
    int l = trunc(log2(procs));
    int rows_pow = l/2 + l%2;
    int cols_pow = l/2;
    rows = rows_pow ? 2<<(rows_pow-1) : 1; // 2^rows_pow
    cols = cols_pow ? 2<<(cols_pow-1) : 1; // 2^cols_pow
}

int main(int argc, char **argv)
{
    // Parse args
    if (argc >= 3) {
        int check = sscanf(argv[1], "%d", &N1);
        N1 = check ? N1 : N1_DEFAULT;
        check = sscanf(argv[2], "%d", &N2);
        N2 = check ? N2 : N2_DEFAULT;
    } else {
        N1 = N1_DEFAULT;
        N2 = N2_DEFAULT;
    }

    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    int rows, cols;
    distribute_procs(procs, rows, cols);
    if (!rank)
        printf("ROWS %d COLS %d\n", rows, cols);

    Node me(rank, rows, cols);

    me.Print();

    MPI_Finalize();
    return 0;
}
