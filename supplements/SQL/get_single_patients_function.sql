create or replace function get_single_patients(selected_icustay_id integer)		
returns table (
		icustay_id				integer,
		itemid					integer,
		label					varchar(200),
		charttime				timestamp without time zone,
		cgid					integer,		-- needed ??
		value					varchar(255),
		valueuom				varchar(50),
		warning					integer,
		error					integer,
		resultstatus			varchar(50),
		stopped					varchar(50)
) 
language plpgsql
as $body$
declare
-- variable declaration with ;	
	
begin
	create or replace view single_icustay_events
	AS
		with d_items as (		-- get labels for event items
		Select
			d_items.itemid,
			d_items.label
		FROM mimiciii.d_items
		), events as
		(
		SELECT
			*
		FROM mimiciii.chartevents
		WHERE icustay_id = selected_icustay_id		-- only filter in For-Loop with icustay_id
		-- AND itemid IN (SELECT * FROM new_table_with_relevant_lables_itemids)			
			-- only use relevant item_ids, otherwise 12.000 different possible chart_events
			-- probably first step of filtering here: database -> not carevue?
		)
		SELECT 
			events.icustay_id,
			events.itemid,
			d_items.label,
			events.charttime,
			events.cgid,		-- needed ??
			events.value,
			events.valueuom,
			events.warning,
			events.error,
			events.resultstatus,
			events.stopped
		FROM events
		LEFT JOIN d_items ON events.itemid = d_items.itemid
		-- where error = none or not events.stopped = none		-- check how to filter this
		order by events.icustay_id, events.itemid, events.charttime;

	RETURN QUERY
	Select * FROM public.single_icustay_events where single_icustay_events.icustay_id = selected_icustay_id;

end; $body$
