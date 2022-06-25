#include <iostream>
#include <vector>
using namespace std;

vector<int> t;

int greatest_common_divisor(int a, int b)
{
    int t = max(a, b);
    b = min(a, b);
    a = t;

    if (b == 0)
    {
        return a;
    }

    if (a % b == 0)
    {
        return b;
    }

    return greatest_common_divisor(a % b, b);
}

void set(int i, int v, int x, int xl, int xr)
{
    if (xl == xr - 1)
    {
        t[x] = v;
        return;
    }

    int xm = (xl + xr) / 2;

    if (i < xm)
    {
        set(i, v, 2 * x + 1, xl, xm);
    }
    else
    {
        set(i, v, 2 * x + 2, xm, xr);
    }
    t[x] = greatest_common_divisor(t[2 * x + 1], t[2 * x + 2]);
}

int gcd(int l, int r, int x, int xl, int xr)
{
    if (xl >= r || l >= xr)
    {
        return 0;
    }
    if (xl >= l && xr <= r)
    {
        return t[x];
    }
    int xm = (xl + xr) / 2;
    int sl = gcd(l, r, 2 * x + 1, xl, xm);
    int sr = gcd(l, r, 2 * x + 2, xm, xr);

    return greatest_common_divisor(sl, sr);
}

int main()
{
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++)
    {
        cin >> a[i];
    }

    int power = 1;
    while (n > power)
    {
        power <<= 1;
    }

    for (int i = 0; i < 2 * power - 1; i++)
    {
        if (i >= (power - 1) && i < (power - 1 + n))
        {
            t.push_back(a[i - (power - 1)]);
        }
        else
        {
            t.push_back(0);
        }
    }

    for (int i = t.size() - 1; i > 0; i -= 2)
    {
        if (i % 2 == 0)
        {
            t[(i - 1) / 2] = greatest_common_divisor(t[i], t[i - 1]);
        }
    }

    int m;
    cin >> m;
    for (int i = 0; i < m; i++)
    {
        int q, l, r;
        cin >> q >> l >> r;
        l -= 1;
        if (q == 1)
        {
            cout << gcd(l + (power - 1), r + (power - 1), 0, power - 1, 2 * power - 1) << endl;
        }

        if (q == 2)
        {
            set(l + (power - 1), r, 0, power - 1, 2 * power - 1);
        }
    }

    return 0;
};