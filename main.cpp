// Catherine Galkina, group 624, year 2017

#include <math.h>
#include <stdio.h>
#include <mpi.h>
//#include <omp.h>

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

inline double F(double x, double y)
{
    double xx = x*x, yy = y*y;
    return 2*(xx + yy)*(1 - 2*xx*yy)*exp(1 - xx*yy);
}

inline double phi(double x, double y)
{
    double xx = x*x, yy = y*y;
    return exp(1 - xx*yy);
}

inline double f_node(double t)
{
    return (pow(1 + t, q) - 1)/(pow(2.0, q) - 1);
}

inline double x_i(int i)
{
    if (i < 0 || i > N1) return -1;
    double f = f_node((double)i/N1);
    return A2*f + A1*(1 - f);
}

inline double y_j(int j)
{
    if (j < 0 || j > N2) return -1;
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
    int my_i;
    // Row
    int my_j;

    // Ranks of neighbours
    int left;
    int right;
    int up;
    int down;

    // Points distribution
    int x1;
    int x2;
    int nx;
    int y1;
    int y2;
    int ny;
    double *xs;
    double *ys;
    double **p;
    double **p_prev;
    double **r;
    double **g;
    double **l;

    double *row_buf;
    double *col_buf;

    int step;

    inline bool border_point(int i, int j)
    {
        return (i == 0 || j == 0 || i == N1 || j == N2) && !fake_point(i, j);
    }

    inline bool fake_point(int i, int j)
    {
        return (i < 0 || j < 0 || i > N1 || j > N2);
    }

    inline double h1(int i) { return (xs[i+1-x1+1] - xs[i-x1+1]); }
    inline double h2(int j) { return (ys[j+1-y1+1] - ys[j-y1+1]); }
    inline double h1_(int i) { return 0.5*(h1(i) + h1(i-1)); }
    inline double h2_(int j) { return 0.5*(h2(j) + h2(j-1)); }

    double dot(int from_i, int to_i, int from_j, int to_j,
               double **u, double **v)
    {
        double sum = 0;
        int i, j;
        //#pragma omp parallel for reduction(+:sum)
        for (i = from_i; i <= to_i; i++) {
            int ii = i - x1 + 1;
            for (j = from_j; j <= to_j; j++) {
                int jj = j - y1 + 1;
                if (!border_point(i, j) && !fake_point(i, j))
                    sum += h1_(i)*h2_(j)*u[ii][jj]*v[ii][jj];
            }
        }
        return sum;
    }

    double comm_dot(int from_i, int to_i, int from_j, int to_j,
               double **u, double **v)
    {
        double my_dot = dot(from_i, to_i, from_j, to_j, u, v);
        double sum;
        MPI_Allreduce(&my_dot, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return sum;
    }

    inline double laplace_elem(int i, int j, double **f)
    {
        if (border_point(i, j) || fake_point(i, j))
            return 0;
        int ii = i - x1 + 1;
        int jj = j - y1 + 1;
        double fij = f[ii][jj];
        double fim1j = f[ii-1][jj];
        double fip1j = f[ii+1][jj];
        double fijm1 = f[ii][jj-1];
        double fijp1 = f[ii][jj+1];
        double f1 = (fij - fim1j)/h1(i-1) - (fip1j - fij)/h1(i);
        double f2 = (fij - fijm1)/h2(j-1) - (fijp1 - fij)/h2(j);
        return f1/h1_(i) + f2/h2_(j);
    }

    void laplace(int from_i, int to_i, int from_j, int to_j, double **f)
    {
        int i, j, ii, jj;
        //#pragma omp parallel for private(ii, jj)
        for (i = from_i; i <= to_i; i++) {
            ii = i-x1+1;
            for (j = from_j; j <= to_j; j++) {
                jj = j-y1+1;
                l[ii][jj] = laplace_elem(i, j, f);
            }
        }
    }

    void row_from_buf(int j, double **m) {
        //#pragma omp parallel for
        for (int i = x1; i <= x2; i++)
            m[i - x1 + 1][j - y1 + 1] = row_buf[i - x1 + 1];
    }

    void row_to_buf(int j, double **m) {
        //#pragma omp parallel for
        for (int i = x1; i <= x2; i++)
            row_buf[i - x1 + 1] = m[i - x1 + 1][j - y1 + 1];
    }

    void col_from_buf(int i, double **m) {
        //#pragma omp parallel for
        for (int j = y1; j <= y2; j++)
            m[i - x1 + 1][j - y1 + 1] = col_buf[j - y1 + 1];
    }

    void col_to_buf(int i, double **m) {
        //#pragma omp parallel for
        for (int j = y1; j <= y2; j++)
            col_buf[j - y1 + 1] = m[i - x1 + 1][j - y1 + 1];
    }

public:

    int GetStep() { return step; }

    double Dev()
    {
        int i, j, ii, jj;
        for (i = x1-1; i <= x2+1; i++) {
            ii = i - x1 + 1;
            for (j = y1-1; j <= y2+1; j++) {
                jj = j - y1 + 1;
                if (fake_point(i, j) ||
                    i == x1-1 || i == x2+1 || j == y1-1 || i == y2+1)
                    p[ii][jj] = 0;
                else
                    p[ii][jj] -= phi(xs[ii], ys[jj]);
            }
        }
        return dot(x1, x2, y1, y2, p, p);
    }

    Node(int rn, int rows, int cols)
    {
        rank = rn;
        int j = rn/cols;
        int i = rn - j*cols;
        my_i = i;
        my_j = j;

        int left_i = (i-1 >= 0) ? i-1 : cols - 1;
        int right_i = (i+1)%cols;
        int up_j = (j-1 >= 0) ? j-1 : rows - 1;
        int down_j = (j+1)%rows;
        left = (rank%cols) ? rank - 1 : -1;
        right = ((rank+1)%cols) ? rank + 1 : -1;
        up = (rank-cols >= 0) ? rank - cols : -1;
        down = (rank+cols < rows*cols) ? rank + cols : -1;

        distribute_points(N1+1, i, cols, x1, x2);
        distribute_points(N2+1, j, rows, y1, y2);
        nx = x2 - x1 + 1;
        ny = y2 - y1 + 1;

        step = 0;
        xs = NULL;
        ys = NULL;
        p = NULL;
        p_prev = NULL;
        r = NULL;
        g = NULL;
        l = NULL;
        row_buf = NULL;
        col_buf = NULL;
    }

    void Init()
    {
        int i, j, ii, jj;
        // + 2 is for neighbours
        xs = new double [nx+2];
        ys = new double [ny+2];
        //#pragma omp parallel for
        for (i = x1-1; i <= x2+1; i++)
            xs[i-x1+1] = x_i(i);
        //#pragma omp parallel for
        for (j = y1-1; j <= y2+1; j++)
            ys[j-y1+1] = y_j(j);

        p_prev = new double*[nx+2];
        //#pragma omp parallel for private(ii, jj)
        for (i = x1-1; i <= x2+1; i++) {
            ii = i - x1 + 1;
            p_prev[ii] = new double[ny+2];
            for (j = y1-1; j <= y2+1; j++) {
                jj = j - y1 + 1;
                if (border_point(i, j))
                    p_prev[ii][jj] = phi(xs[ii], ys[jj]);
                else
                    p_prev[ii][jj] = 0;
            }
        }

        p = new double*[nx+2];
        //#pragma omp parallel for private(ii, jj)
        for (i = x1-1; i <= x2+1; i++) {
            ii = i - x1 + 1;
            p[ii] = new double[ny+2];
            for (j = y1-1; j <= y2+1; j++) {
                jj = j - y1 + 1;
                p[ii][jj] = 0;
            }
        }

        r = new double*[nx+2];
        //#pragma omp parallel for private(ii, jj)
        for (i = x1-1; i <= x2+1; i++) {
            ii = i - x1 + 1;
            r[ii] = new double[ny+2];
            for (j = y1-1; j <= y2+1; j++) {
                jj = j - y1 + 1;
                if (border_point(i, j) || fake_point(i, j) ||
                    i == x1-1 || i == x2+1 || j == y1-1 || i == y2+1) // because of laplace
                    r[ii][jj] = 0;
                else
                    r[ii][jj] =
                        laplace_elem(i, j, p_prev) - F(xs[ii], ys[jj]);
            }
        }

        g = new double*[nx+2];
        //#pragma omp parallel for private(ii, jj)
        for (i = x1-1; i <= x2+1; i++) {
            ii = i - x1 + 1;
            g[ii] = new double[ny+2];
            for (j = y1-1; j <= y2+1; j++) {
                jj = j - y1 + 1;
                g[ii][jj] = r[ii][jj];
            }
        }

        l = new double*[nx+2];
        //#pragma omp parallel for private(ii, jj)
        for (i = x1-1; i <= x2+1; i++) {
            ii = i - x1 + 1;
            l[ii] = new double[ny+2];
            for (j = y1-1; j <= y2+1; j++) {
                jj = j - y1 + 1;
                l[ii][jj] = 0;
            }
        }

        row_buf = new double[nx+2];
        col_buf = new double[ny+2];
    }

    double Step()
    {
        step++;
        int i, j, ii, jj;
        double tau;

        // Iteration step
        if (step == 1) {
            double dot1 = comm_dot(x1, x2, y1, y2, r, r);
            Exchange(r);
            laplace(x1, x2, y1, y2, r);
            double dot2 = comm_dot(x1, x2, y1, y2, l, r);
            tau = dot1/dot2;
            //#pragma omp parallel for private(ii, jj)
            for (i = x1; i <= x2; i++) {
                ii = i-x1+1;
                for (j = y1; j <= y2; j++) {
                    jj = j-y1+1;
                    p[ii][jj] = p_prev[ii][jj] - tau*r[ii][jj];
                }
            }
        } else {
            Exchange(r);
            laplace(x1, x2, y1, y2, r);
            double dot1 = comm_dot(x1, x2, y1, y2, l, g);
            Exchange(g);
            laplace(x1, x2, y1, y2, g);
            double dot2 = comm_dot(x1, x2, y1, y2, l, g);
            double alpha = dot1/dot2;
            //#pragma omp parallel for private(ii, jj)
            for (i = x1; i <= x2; i++) {
                ii = i-x1+1;
                for (j = y1; j <= y2; j++) {
                    jj = j-y1+1;
                    g[ii][jj] = r[ii][jj] - alpha*g[ii][jj];
                }
            }
            dot1 = comm_dot(x1, x2, y1, y2, r, g);
            Exchange(g);
            laplace(x1, x2, y1, y2, g);
            dot2 = comm_dot(x1, x2, y1, y2, l, g);
            tau = dot1/dot2;
            //#pragma omp parallel for private(ii, jj)
            for (i = x1; i <= x2; i++) {
                ii = i-x1+1;
                for (j = y1; j <= y2; j++) {
                    jj = j-y1+1;
                    p[ii][jj] = p_prev[ii][jj] - tau*g[ii][jj];
                }
            }
        }

        // Calculate error
        for (i = x1; i <= x2; i++) {
            ii = i-x1+1;
            for (j = y1; j <= y2; j++) {
                jj = j-y1+1;
                p_prev[ii][jj] = p[ii][jj] - p_prev[ii][jj];
            }
        }
        double err = sqrt(comm_dot(x1, x2, y1, y2, p_prev, p_prev));

        // Save p to p_prev
        //#pragma omp parallel for private(ii, jj)
        for (i = x1; i <= x2; i++) {
            ii = i-x1+1;
            for (j = y1; j <= y2; j++) {
                jj = j-y1+1;
                p_prev[ii][jj] = p[ii][jj];
            }
        }

        Exchange(p_prev);

        //#pragma omp parallel for private(ii, jj)
        for (i = x1; i <= x2; i++) {
            ii = i-x1+1;
            for (j = y1; j <= y2; j++) {
                jj = j-y1+1;
                if (border_point(i,j) || fake_point(i,j))
                  r[ii][jj] = 0;
                else
                  r[ii][jj] = laplace_elem(i, j, p_prev) - F(xs[ii], ys[jj]);
            }
        }

        return err;
    }

    void Exchange(double **m)
    {
        MPI_Status status;

        // Get row y1-1 from up
        if (up != -1) {
            MPI_Recv(row_buf, nx+2, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &status);
            row_from_buf(y1-1, m);
        }
        // Get column x1-1 from left
        if (left != -1) {
            MPI_Recv(col_buf, ny+2, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &status);
            col_from_buf(x1-1, m);
        }
        // Send row y2 to down
        if (down != -1) {
            row_to_buf(y2, m);
            MPI_Send(row_buf, nx+2, MPI_DOUBLE, down, 0, MPI_COMM_WORLD);
        }
        // Send column x2 to right
        if (right != -1) {
            col_to_buf(x2, m);
            MPI_Send(col_buf, ny+2, MPI_DOUBLE, right, 0, MPI_COMM_WORLD);
        }
        // Send row x1 to up
        if (up != -1) {
            row_to_buf(x1, m);
            MPI_Send(row_buf, nx+2, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);
        }
        // Send column x1 to left
        if (left != -1) {
            col_to_buf(x1, m);
            MPI_Send(col_buf, ny+2, MPI_DOUBLE, left, 0, MPI_COMM_WORLD);
        }
        // Get row y2+1 from down
        if (down != -1) {
            MPI_Recv(row_buf, nx+2, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &status);
            row_from_buf(y2+1, m);
        }
        // Get column x2+1 from right
        if (right != -1) {
            MPI_Recv(col_buf, ny+2, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &status);
            col_from_buf(x2+1, m);
        }
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
        printf(format, rank, my_i, rank, my_j, rank, left,
                rank, right, rank, up, rank, down,
                rank, x1, x2,
                rank, y1, y2);
        for (int i = x1 - 1; i <= x2 + 1; i++)
            for (int j = y1 - 1; j <= y2 + 1; j++)
                printf("%d %d %d %d %f\n", rank, i, j,
                        border_point(i, j), p_prev[i-x1+1][j-y1+1]);
    }

    ~Node() {
        if (xs)
            delete [] xs;
        if (ys)
            delete [] ys;
        if (row_buf)
            delete [] row_buf;
        if (col_buf)
            delete [] col_buf;
        int i;
        if (p) {
            for (i = 0; i < nx+2; i++)
                delete [] p[i];
            delete [] p;
        }
        if (p_prev) {
            for (i = 0; i < nx+2; i++)
                delete [] p_prev[i];
            delete [] p_prev;
        }
        if (r) {
            for (i = 0; i < nx+2; i++)
                delete [] r[i];
            delete [] r;
        }
        if (g) {
            for (i = 0; i < nx+2; i++)
                delete [] g[i];
            delete [] g;
        }
        if (l) {
            for (i = 0; i < nx+2; i++)
                delete [] l[i];
            delete [] l;
        }
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
    double t1, t2;
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    me.Init(); // step 0
    //me.Print();

    double err;
    do {
        err = me.Step();
        //if (!rank) printf("Err = %f\n", err);
    } while (err >= eps);
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    double my_dt = t2 - t1;
    double max_dt;
    MPI_Reduce(&my_dt, &max_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double my_dev = me.Dev(), dev;
    MPI_Reduce(&my_dev, &dev, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank)
        printf("Max time = %f\nSteps = %d\nDev = %f\n", max_dt, me.GetStep(), dev);
    MPI_Finalize();
    return 0;
}
