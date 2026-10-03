-- Query 1: Basic selection
SELECT source_row_id, make, model, year, engine_hp, msrp
FROM vehicles
ORDER BY source_row_id
LIMIT 10;

-- Query 2: Filtering
SELECT source_row_id, make, model, year, msrp
FROM vehicles
WHERE year >= 2015
ORDER BY source_row_id
LIMIT 10;

-- Query 3: Multiple conditions
SELECT source_row_id, make, model, year, engine_hp, msrp
FROM vehicles
WHERE year >= 2015 AND engine_hp >= 300 AND msrp < 40000
ORDER BY msrp ASC, source_row_id ASC
LIMIT 10;

-- Query 4: Sorting
SELECT source_row_id, make, model, year, engine_hp, msrp
FROM vehicles
WHERE msrp IS NOT NULL
ORDER BY msrp DESC, source_row_id ASC
LIMIT 10;

-- Query 5: Aggregation
SELECT COUNT(*) AS records,
       MIN(year) AS earliest_year, MAX(year) AS latest_year,
       COUNT(msrp) AS price_records,
       ROUND(AVG(msrp), 2) AS avg_msrp,
       COUNT(engine_hp) AS hp_records,
       ROUND(AVG(engine_hp), 2) AS avg_hp
FROM vehicles;

-- Query 6: Group comparison
SELECT vehicle_size, COUNT(*) AS records,
       ROUND(AVG(msrp), 2) AS avg_msrp
FROM vehicles
WHERE vehicle_size IS NOT NULL AND msrp IS NOT NULL
GROUP BY vehicle_size
ORDER BY avg_msrp DESC, vehicle_size ASC;

-- Query 7: HAVING
SELECT make, COUNT(*) AS records,
       ROUND(AVG(msrp), 2) AS avg_msrp
FROM vehicles
WHERE make IS NOT NULL AND msrp IS NOT NULL
GROUP BY make
HAVING COUNT(*) >= 100
ORDER BY avg_msrp DESC, make ASC;

-- Query 8: JOIN
SELECT t.transmission_name, COUNT(*) AS records,
       ROUND(AVG(v.highway_mpg), 2) AS avg_highway_mpg
FROM vehicles AS v
JOIN transmissions AS t ON v.transmission_id = t.transmission_id
WHERE v.fuel_economy_analysis_ready = 1
GROUP BY t.transmission_id, t.transmission_name
ORDER BY avg_highway_mpg DESC, t.transmission_name ASC;
