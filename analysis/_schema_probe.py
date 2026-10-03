import sqlite3, sys
p = sys.argv[1] if len(sys.argv) > 1 else "data/benchmarks/longmemeval_banks/asem_thg.sqlite"
c = sqlite3.connect("file:" + p + "?mode=ro", uri=True)
for (n, s) in c.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table','index')"):
    print("==", n)
    print((s or "")[:500])
print("--- counts ---")
for (n,) in c.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    try:
        print(n, c.execute('SELECT COUNT(*) FROM "%s"' % n).fetchone()[0])
    except Exception as e:
        print(n, "ERR", e)
