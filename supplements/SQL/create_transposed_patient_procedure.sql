-- https://stackoverflow.com/questions/12879672/dynamically-generate-columns-for-crosstab-in-postgresql

create or replace procedure public.create_transposed_patient(input_icustay_id integer, selected_itemids_list varchar[])		
language plpgsql
as $body$
declare
	rec RECORD;
	v_patient_record RECORD;
	str text;
	
begin
	raise notice 'Creating transposed_patient for icustay_id: %', input_icustay_id;

-- Step 1: Get all_events_view temporarily into temp_single_patient table
	DROP TABLE IF EXISTS public.temp_single_patient;				
    CREATE TABLE public.temp_single_patient AS
    SELECT * FROM public.get_all_events_view(input_icustay_id, selected_itemids_list);		-- selected_itemids_list can be '{}', then all available itemids will be selected

	str := ' "charttime" timestamp without time zone, ';			-- text = field type , not icustay_id in here, because it would group all rows
   	FOR rec IN SELECT DISTINCT label								-- looping to get column heading string
        FROM temp_single_patient									
		ORDER BY temp_single_patient.label
	LOOP
    	str :=  str || '"' || rec.label || '" text' ||',';			-- str is used for defintion of table below
    END LOOP;
    str:= substring(str, 0, length(str));							-- remove last comma
	raise notice 'Selected labels for header str: %', str;

-- Step 2: Dynamically create a temp_table with transposed row->column for selected labels
    EXECUTE 												 
	'CREATE EXTENSION IF NOT EXISTS tablefunc;
    DROP TABLE IF EXISTS temp_transposed_patient;
    CREATE TABLE temp_transposed_patient AS
    SELECT *
    FROM crosstab(''select temp_single_patient.charttime, temp_single_patient.label, temp_single_patient.value 
					from temp_single_patient order by 1'',
                 ''SELECT DISTINCT temp_single_patient.label FROM temp_single_patient ORDER BY 1'')
    AS final_result ('|| str ||')';
	
-- Setp 3: Join the chart_events data with the general patient data
	DELETE FROM temp_transposed_patient WHERE temp_transposed_patient.charttime = '0001-01-01 00:00:01';	-- remove DUMMY values, those were added in get_all_events_view
	
	-- check if not exists (should never be the case, always call single transposed_patient after having created temp_filtered_patient_cohort)
	if not (SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename  = 'temp_filtered_patient_cohort')) then	
		raise notice 'WARNING: temp_filtered_patient_cohort should have already existed for creation of transposed_patient.';
		CREATE TABLE public.temp_filtered_patient_cohort AS
		SELECT * FROM get_filtered_patient_cohort('{}');		
	end if;
	
	SELECT * FROM temp_filtered_patient_cohort WHERE temp_filtered_patient_cohort.icustay_id = input_icustay_id INTO v_patient_record;
	ALTER TABLE public.temp_transposed_patient
		ADD COLUMN icustay_id					integer,
		ADD COLUMN hadm_id						integer, 
		ADD COLUMN subject_id						integer, 
		ADD COLUMN intime						timestamp without time zone, 
		ADD COLUMN outtime						timestamp without time zone, 
		ADD COLUMN los_hours 					numeric,
		ADD COLUMN icustays_count				bigint,
		ADD COLUMN age							numeric,
		ADD COLUMN patientweight				double precision,
		ADD COLUMN gender						varchar(5),
		ADD COLUMN ethnicity					varchar(200),
		ADD COLUMN admission_type				varchar(50),
		ADD COLUMN discharge_location 			varchar(50),
		ADD COLUMN insurance 					varchar(255),
		ADD COLUMN language 					varchar(10),
		ADD COLUMN religion						varchar(50),
		ADD COLUMN marital_status				varchar(50),
		ADD COLUMN diagnosis_text				text,
		ADD COLUMN dob 							date,
		ADD COLUMN dod 							date,
		ADD COLUMN death_in_hosp 				int,
		ADD COLUMN death_3_days 				int,
		ADD COLUMN death_30_days 				int,
		ADD COLUMN death_180_days 				int,
		ADD COLUMN death_365_days 				int,
		ADD COLUMN icd9_code					varchar(10),
		ADD COLUMN stroke_type					text,
		ADD COLUMN all_icd9_codes 				varchar[];											
	UPDATE public.temp_transposed_patient 
	SET 
		icustay_id = input_icustay_id,
		hadm_id = v_patient_record.hadm_id,
		subject_id = v_patient_record.subject_id,
		intime = v_patient_record.intime,
		outtime = v_patient_record.outtime,
		los_hours = v_patient_record.los_hours,
		icustays_count = v_patient_record.icustays_count,
		age = v_patient_record.age,
		patientweight = v_patient_record.patientweight,
		gender = v_patient_record.gender,
		ethnicity = v_patient_record.ethnicity,
		admission_type = v_patient_record.admission_type,
		discharge_location = v_patient_record.discharge_location,
		insurance = v_patient_record.insurance,
		language = v_patient_record.language,
		religion = v_patient_record.religion,
		marital_status = v_patient_record.marital_status,
		diagnosis_text = v_patient_record.diagnosis_text,
		dob = v_patient_record.dob,	
		dod = v_patient_record.dod,						
		death_in_hosp = v_patient_record.death_in_hosp, 				
		death_3_days = v_patient_record.death_3_days, 				
		death_30_days = v_patient_record.death_30_days,  
		death_180_days = v_patient_record.death_180_days, 
		death_365_days = v_patient_record.death_365_days, 				
		icd9_code = v_patient_record.icd9_code,
		stroke_type = v_patient_record.stroke_type,
		all_icd9_codes = v_patient_record.all_icd9_codes;

end; $body$
