---inner join 2

SELECT c.code AS country_code, c.name, e.year, e.inflation_rate
FROM countries AS c INNER JOIN economies AS e
    -- 2. Match on code
    ON c.code = e.code;

---inner join(3)


SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate

  FROM countries AS c INNER JOIN populations AS p
    
    ON c.code = p.country_code
  
  INNER JOIN economies AS e
  
    ON c.code = e.code;

---Inner join (3)
-- 6. Select fields
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
  -- 1. From countries (alias as c), -- 2. Join to populations (as p)
  FROM countries AS c INNER JOIN populations AS p
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

--- INNER Challenge
SELECT country_code, size,
  CASE WHEN size > 50000000
            THEN 'large'
       WHEN size > 1000000
            THEN 'medium'
       ELSE 'small' END
       AS popsize_group
INTO pop_plus       
FROM populations
WHERE year = 2015;


SELECT name, continent, geosize_group, popsize_group

FROM countries_plus AS c JOIN pop_plus AS p

    ON c.code = p.country_code

ORDER BY geosize_group;



--- CHAPTER2 - Outer joins and cross joins
SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
-- 1. Join right table (with alias)
FROM cities AS c1 LEFT JOIN countries AS c2
  -- 2. Match on country code
    ON c1.country_code = c2.code
-- 3. Order by descending country code
ORDER BY code DESC;

/*
5. Select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c) -- 2. Join to right table (alias as l)
FROM countries AS c LEFT JOIN languages AS l
  
  
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;

--LEFT JOIN
 -- Select fields
SELECT region, AVG(gdp_percapita) AS avg_gdp
-- From countries (alias as c) -- Left join with economies (alias as e)
FROM countries as c LEFT JOIN economies AS e
    -- Match on code fields
    ON c.code = e.code
-- Focus on 2010
WHERE year = 2010
-- Group by region
GROUP BY  region
-- Order by descending avg_gdp
ORDER BY avg_gdp DESC;

--RIGHT JOIN
SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM languages
  RIGHT JOIN countries
    ON languages.code = countries.code
  RIGHT JOIN cities
    ON countries.code = cities.country_code
ORDER BY city, language;
--FULL JOIN
SELECT name AS country, code, region, basic_unit
-- 3. From countries
FROM countries
  -- 4. Join to currencies
  INNER JOIN currencies
    -- 5. Match on code
    USING (code)
-- 1. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 2. Order by region
ORDER BY region;