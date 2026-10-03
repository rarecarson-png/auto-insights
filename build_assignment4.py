"""Rebuild SQLite database and query CSVs from Assignment 3, using Python's standard library."""
from pathlib import Path
import csv, sqlite3, hashlib, json

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data_cleaned.csv'
DB = ROOT / 'auto_insights.sqlite'
QUERIES = [
('Basic selection', 'Inspect ten vehicle records and the fields used in the price analysis.', '''SELECT source_row_id, make, model, year, engine_hp, msrp
FROM vehicles
ORDER BY source_row_id
LIMIT 10;'''),
('Filtering', 'Find a reproducible example of newer model-year records. The ten displayed rows are a sample, not all matches.', '''SELECT source_row_id, make, model, year, msrp
FROM vehicles
WHERE year >= 2015
ORDER BY source_row_id
LIMIT 10;'''),
('Multiple conditions', 'Find recent-model-year records combining at least 300 horsepower with a recorded MSRP below $40,000.', '''SELECT source_row_id, make, model, year, engine_hp, msrp
FROM vehicles
WHERE year >= 2015 AND engine_hp >= 300 AND msrp < 40000
ORDER BY msrp ASC, source_row_id ASC
LIMIT 10;'''),
('Sorting', 'Identify the ten highest recorded prices for inspection. High prices remain in the data.', '''SELECT source_row_id, make, model, year, engine_hp, msrp
FROM vehicles
WHERE msrp IS NOT NULL
ORDER BY msrp DESC, source_row_id ASC
LIMIT 10;'''),
('Aggregation', 'Summarize dataset coverage and available-case averages. COUNT(column) shows each numeric denominator.', '''SELECT COUNT(*) AS records,
       MIN(year) AS earliest_year, MAX(year) AS latest_year,
       COUNT(msrp) AS price_records,
       ROUND(AVG(msrp), 2) AS avg_msrp,
       COUNT(engine_hp) AS hp_records,
       ROUND(AVG(engine_hp), 2) AS avg_hp
FROM vehicles;'''),
('Group comparison', 'Compare average recorded MSRP across vehicle sizes. This result supplies Visualization 1.', '''SELECT vehicle_size, COUNT(*) AS records,
       ROUND(AVG(msrp), 2) AS avg_msrp
FROM vehicles
WHERE vehicle_size IS NOT NULL AND msrp IS NOT NULL
GROUP BY vehicle_size
ORDER BY avg_msrp DESC, vehicle_size ASC;'''),
('HAVING', 'Compare every brand with at least 100 priced records. The threshold avoids the smallest groups but does not guarantee statistical reliability.', '''SELECT make, COUNT(*) AS records,
       ROUND(AVG(msrp), 2) AS avg_msrp
FROM vehicles
WHERE make IS NOT NULL AND msrp IS NOT NULL
GROUP BY make
HAVING COUNT(*) >= 100
ORDER BY avg_msrp DESC, make ASC;'''),
('JOIN', 'Join vehicle records to a transmission lookup, then compare average highway MPG in the eligible fuel-economy subset. This result supplies Visualization 2.', '''SELECT t.transmission_name, COUNT(*) AS records,
       ROUND(AVG(v.highway_mpg), 2) AS avg_highway_mpg
FROM vehicles AS v
JOIN transmissions AS t ON v.transmission_id = t.transmission_id
WHERE v.fuel_economy_analysis_ready = 1
GROUP BY t.transmission_id, t.transmission_name
ORDER BY avg_highway_mpg DESC, t.transmission_name ASC;''')
]

def build():
    with SOURCE.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        records = list(reader)
    text_columns = {'make','model','fuel_type','transmission','driven_wheels','market_category','vehicle_size','vehicle_style'}
    names = sorted({r['transmission'] for r in records if r['transmission']})
    transmission_ids = {name: i+1 for i, name in enumerate(names)}
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.execute('PRAGMA foreign_keys = ON')
    declarations = []
    for col in columns:
        if col == 'transmission':
            declarations.append('transmission_id INTEGER REFERENCES transmissions(transmission_id)')
        elif col == 'source_row_id':
            declarations.append('source_row_id INTEGER PRIMARY KEY')
        else:
            kind = 'TEXT' if col in text_columns else 'REAL' if col == 'log10_msrp' else 'INTEGER'
            declarations.append(f'{col} {kind}')
    schema = ('PRAGMA foreign_keys = ON;\nCREATE TABLE transmissions (\n'
              '  transmission_id INTEGER PRIMARY KEY,\n  transmission_name TEXT NOT NULL UNIQUE\n);\n'
              'CREATE TABLE vehicles (\n  ' + ',\n  '.join(declarations) + '\n);\n')
    con.executescript(schema)
    con.executemany('INSERT INTO transmissions VALUES (?, ?)', [(i,n) for n,i in transmission_ids.items()])
    def convert(col, value):
        if col == 'transmission': return transmission_ids.get(value)
        if value == '': return None
        if col in text_columns: return value
        if value in ('True', 'False'): return int(value == 'True')
        return float(value) if col == 'log10_msrp' else int(value)
    con.executemany('INSERT INTO vehicles VALUES (' + ','.join('?' for _ in columns) + ')',
                    [[convert(col, r[col]) for col in columns] for r in records])
    con.commit()
    assert con.execute('SELECT COUNT(*) FROM vehicles').fetchone()[0] == len(records) == 11199
    assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert not con.execute('PRAGMA foreign_key_check').fetchall()
    assert con.execute('SELECT COUNT(*) FROM vehicles v LEFT JOIN transmissions t ON v.transmission_id=t.transmission_id').fetchone()[0] == len(records)
    (ROOT/'schema.sql').write_text(schema)
    (ROOT/'assignment4_queries.sql').write_text('\n\n'.join(f'-- Query {i}: {title}\n{sql}' for i,(title,_,sql) in enumerate(QUERIES,1))+'\n')
    (ROOT/'sql_results').mkdir(exist_ok=True)
    results = []
    for i, (title, desc, sql) in enumerate(QUERIES,1):
        cursor = con.execute(sql)
        fields = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
        with (ROOT/'sql_results'/f'query_{i:02d}.csv').open('w',newline='') as f:
            writer = csv.writer(f); writer.writerow(fields); writer.writerows(rows)
        results.append(dict(number=i,title=title,description=desc,sql=sql,fields=fields,rows=rows))
    metadata = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), records=len(records),
                    transmission_categories=len(names), sqlite_version=sqlite3.sqlite_version,
                    unknown_transmissions=con.execute('SELECT COUNT(*) FROM vehicles WHERE transmission_id IS NULL').fetchone()[0])
    (ROOT/'sql_results'/'query_results.json').write_text(json.dumps(dict(metadata=metadata,queries=results),indent=2))
    con.close()
    print(json.dumps(metadata, indent=2))
    return results,metadata

if __name__ == '__main__':
    build()
