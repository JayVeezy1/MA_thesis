create or replace function public.get_all_events_view(input_icustay_id integer, selected_itemids_list varchar[])		
returns table (
		icustay_id				integer,
		itemid					integer,
		label					varchar(200),
		charttime				timestamp without time zone,
		-- cgid					integer,		-- caregiver id - not needed
		value					varchar(255)
		-- valueuom				varchar(50)		-- not useful here because lost when doing transposition, join later in dictionary
) 
language plpgsql
as $body$
declare
v_selected_itemid integer;
temp_related_label varchar(200);
	
begin
	-- 0. step create list from selected itemids input string
	if array_length(selected_itemids_list, 1) > 0 then		-- old function: selection of itemids usually not done here but inside python
		if (SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename  = 'temp_selected_itemids')) then	
			DROP TABLE temp_selected_itemids CASCADE;		-- CASCADE needed because view patient_cohort_filtered is related
		end if;

		CREATE TABLE public.temp_selected_itemids(itemids integer);

		-- only get the selected itemids (not used anymore, filtering&selection of itemids/labels will be done in python) 
		FOREACH v_selected_itemid IN ARRAY selected_itemids_list LOOP
			INSERT INTO temp_selected_itemids (itemids)
			VALUES (v_selected_itemid);
		END LOOP;
	end if;

	-- 1. step: merge/join different event types into 'all_events_view'
	CREATE OR REPLACE VIEW public.all_events_view
	AS
		with d_items as (		
		Select
			d_items.itemid,
			d_items.label
		FROM mimiciii.d_items
		), 
		d_labitems as (
			Select
				d_labitems.itemid,
				d_labitems.label
			FROM mimiciii.d_labitems
		),	
		first_icustay_id as (
			SELECT hadm_id, 
				   icustay_id, 
				   intime,	   
			       outtime
			FROM mimiciii.icustays 
			WHERE (hadm_id,icustay_id) IN 
				(SELECT hadm_id, MIN(icustay_id)
				 FROM mimiciii.icustays
				 GROUP BY hadm_id
				 ORDER BY hadm_id)
		), 
		labevents_with_icustay as (
			Select
				first_icustay_id.icustay_id,
				labevents.itemid,
				labevents.charttime,		 
				labevents.value,
				labevents.valueuom
			FROM mimiciii.labevents
			left join first_icustay_id on first_icustay_id.hadm_id = labevents.hadm_id
			where first_icustay_id.outtime >= labevents.charttime
			and first_icustay_id.intime <= labevents.charttime 		-- only acceppt labitems with icustay_id-intime < charttimes < icustay_id-outtime
		),	
		events as
		(
		SELECT
			chartevents.icustay_id,
			chartevents.itemid,
			chartevents.charttime,
			chartevents.value
			-- chartevents.valueuom
		FROM mimiciii.chartevents
		WHERE chartevents.error = 0 
			OR chartevents.warning = 0
			OR not chartevents.resultstatus IS NULL 
			OR not chartevents.stopped IS NULL		
		-- WHERE chartevents.icustay_id = input_icustay_id	-- filtering with parameter here not possible because parameter 'input_icustay_id' is not known inside 'create view...'
		UNION ALL SELECT
			procedureevents_mv.icustay_id,
			procedureevents_mv.itemid,
			procedureevents_mv.starttime as charttime,
			cast(procedureevents_mv.value as character varying(255)) as value
			-- cast(procedureevents_mv.valueuom as character varying(50)) as value
		FROM mimiciii.procedureevents_mv
		UNION ALL SELECT
			inputevents_mv.icustay_id,
			inputevents_mv.itemid,
			inputevents_mv.starttime as charttime,
			cast(inputevents_mv.amount as character varying(255)) as value
			-- cast(inputevents_mv.amountuom as character varying(50)) as value
		FROM mimiciii.inputevents_mv
		UNION ALL SELECT
			labevents_with_icustay.icustay_id,
			labevents_with_icustay.itemid,
			labevents_with_icustay.charttime,
			cast(labevents_with_icustay.value as character varying(255)) as value
			-- cast(labevents_with_icustay.valueuom as character varying(50)) as value
		FROM labevents_with_icustay
		UNION ALL SELECT
			outputevents.icustay_id,
			outputevents.itemid,
			outputevents.charttime,
			cast(outputevents.value as character varying(255)) as value
			-- cast(outputevents.valueuom as character varying(50)) as value
		FROM mimiciii.outputevents
		), 
		first_result as (
		SELECT 
			events.icustay_id,
			events.itemid,
			d_items.label,
			events.charttime,
			events.value
			-- events.valueuom 			
		FROM events
		LEFT JOIN d_items ON events.itemid = d_items.itemid
		)
		SELECT 
			*
		FROM first_result
		order by first_result.icustay_id, first_result.itemid, first_result.charttime;



	-- step 2: add labevent labels into temp_table
	DROP TABLE IF EXISTS public.temp_single_patient_DUMMY;				
    CREATE TABLE public.temp_single_patient_DUMMY AS
    SELECT * FROM all_events_view WHERE all_events_view.icustay_id = input_icustay_id;
	-- add labitems to single_patient
	UPDATE public.temp_single_patient_DUMMY
	SET label = (SELECT d_labitems.label FROM mimiciii.d_labitems WHERE d_labitems.itemid = temp_single_patient_DUMMY.itemid)
	WHERE temp_single_patient_DUMMY.itemid IN (SELECT d_labitems.itemid FROM mimiciii.d_labitems);

	-- step 3: add a dummy value for every selected label, if some were selected (this is not used anymore, label selection will be inside Python)
	-- with charttime 'today' for each label, so the label "exists" and will be turned into a column when doing crosstable
	-- will be removed after doing crossable
	if array_length(selected_itemids_list, 1) > 0 then
		FOREACH v_selected_itemid IN ARRAY selected_itemids_list LOOP

			if v_selected_itemid IN (SELECT d_items.itemid FROM mimiciii.d_items) then
				Select d_items.label from mimiciii.d_items where d_items.itemid = v_selected_itemid INTO temp_related_label;
			elsif v_selected_itemid IN (SELECT d_labitems.itemid FROM mimiciii.d_labitems) then
				Select d_labitems.label from mimiciii.d_labitems where d_labitems.itemid = v_selected_itemid INTO temp_related_label;
			else
				temp_related_label = 'unknown_label';
			end if;

			INSERT INTO temp_single_patient_DUMMY (icustay_id, itemid, charttime, label, value)
			VALUES (input_icustay_id, 
					v_selected_itemid,
					'0001-01-01 00:00:01', 			
					temp_related_label,
					'-1');
		END LOOP;
	end if;
	
	-- step 4: return the events for selected icustay_id and filtered itemids
	if array_length(selected_itemids_list, 1) > 0 then
		RETURN QUERY
		SELECT * FROM temp_single_patient_DUMMY
		WHERE temp_single_patient_DUMMY.itemid IN (SELECT itemids FROM temp_selected_itemids);	-- only use relevant item_ids, otherwise 12.000 different possible chart_events
	else
		RETURN QUERY
		SELECT * FROM temp_single_patient_DUMMY;
	end if;

end; $body$
