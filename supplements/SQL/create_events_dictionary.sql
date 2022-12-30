-- This file creates a table itemid | label | category | dbsource | valueuom for all chosen event-types for this analysis. 
-- The table will be used to connect a label to its category or its unit-of-measurement (valueuom)
-- Other event types like datetimeevents or microbiologyevents might be added further

with chartevents as (
	select 
		distinct(itemid), 
		valueuom
	from mimiciii.chartevents
), 
inputevents_mv as (
	select 
		distinct(itemid), 
		amountuom as valueuom
	from mimiciii.inputevents_mv
),
outputevents as (
	select 
		distinct(itemid), 
		valueuom
	from mimiciii.outputevents
),
procedureevents_mv as (
	select 
		distinct(itemid), 
		valueuom
	from mimiciii.procedureevents_mv
),
-- notevents excluded, cptevents excluded (focus on cost-centers, might be added if also added d_cpt at dictionaries)
-- can also be added, but not yet in current analysis
/*
datetimeevents as (
	select 
		distinct(itemid), 
		valueuom
	from mimiciii.datetimeevents
),
microbiologyevents as (
	select 
		distinct(itemid), 
		valueuom
	from mimiciii.microbiologyevents
),*/
labevents as (
	select 
		distinct(itemid), 
		valueuom
	from mimiciii.labevents
), all_events as(
	select * from chartevents
	UNION ALL SELECT * FROM inputevents_mv
	UNION ALL SELECT * FROM outputevents
	-- UNION ALL SELECT * FROM procedureevents_mv			-- procedures sometimes have same code as diagnosis -> not really needed, removed
	UNION ALL SELECT * FROM labevents
	-- UNION ALL SELECT * FROM datetimeevents				-- can be added, but not yet in current analysis
	-- UNION ALL SELECT * FROM microbiologyevents
), 
-- get dictionaries
d_items as(
	Select 
		d_items.itemid,
		REPLACE(REPLACE(d_items.label, ';', '&'), ',' , '-') as label,
		d_items.category,
		d_items.dbsource
	from mimiciii.d_items
	where not d_items.dbsource = 'carevue'			-- removing carevue, otherwise > 12000 additional itemids
), d_labitems as(
	Select 
		d_labitems.itemid,
		REPLACE(REPLACE(d_labitems.label, ';', '&'), ',' , '-') as label,
		d_labitems.category,
		'labitem' as dbsource						-- labitems have no column dbsource
	from mimiciii.d_labitems
), d_all_items as(
	SELECT * FROM d_items
	UNION ALL SELECT * FROM d_labitems
)
Select *
from (
	select distinct on (d_all_items.itemid)
		d_all_items.itemid,
		d_all_items.label,
		d_all_items.category,
		d_all_items.dbsource,
		all_events.valueuom
	from d_all_items
	left join all_events
		on d_all_items.itemid = all_events.itemid
	order by d_all_items.itemid
) t1
order by t1.itemid;			-- resulting in 4526 itemids
