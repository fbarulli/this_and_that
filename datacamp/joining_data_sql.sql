---inner join 2
-- 3. Select fields with aliases
SELECT c.code AS country_code, c.name, e.year, e.inflation_rate
FROM countries AS c
  -- 1. Join to economies (alias e)
  INNER JOIN economies AS e
    -- 2. Match on code
    ON c.code = e.code;

---inner join(3)

-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code
    ON c.code = e.code;

---Inner join (3)
-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to populations (as p)
  INNER JOIN populations AS p
    -- 3. Match on country code
    ON c.code = p.country_code
  -- 4. Join to economies (as e)
  INNER JOIN economies AS e
    -- 5. Match on country code and year
    ON c.code = e.code
    AND p.year = e.year;

--- INNER JOIN
-- 4. Select fields
SELECT c.name AS country, continent, l.name AS language, l.official
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join to languages (as l)
  INNER JOIN languages as l
    -- 3. Match using code
    USING(code)

-- 5. Select fields with aliases
SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations as p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations as p2
    -- 3. Match on country code
    ON  p1.country_code = p2.country_code
        -- 4. and year (with calculation)
        AND p1.year = p2.year - 5

---Self Join
SELECT p1.country_code,
       p1.size AS size2010, 
       p2.size AS size2015,
       -- 1. calculate growth_perc
       ((p2.size - p1.size)/p1.size * 100.0) AS growth_perc
-- 2. From populations (alias as p1)
FROM populations AS p1
  -- 3. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 4. Match on country code
    ON p1.country_code = p2.country_code
        -- 5. and year (with calculation)
        AND p1.year = p2.year - 5;
--- Case when and then

SELECT name, continent, code, surface_area,
    -- 1. First case
    CASE WHEN surface_area > 2000000 THEN 'large'
        -- 2. Second case
        WHEN  surface_area > 350000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS geosize_group
-- 5. From table
FROM countries;