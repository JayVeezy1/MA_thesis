-- https://stackoverflow.com/questions/12879672/dynamically-generate-columns-for-crosstab-in-postgresql

create or replace procedure create_transposed_patient(input_icustay_id integer, selected_itemids_list varchar[])		
language plpgsql
as $body$
declare
-- variable declaration with ;	
	rec RECORD;
	v_patient_record RECORD;
	str text;
	
begin
-- Step 1: Get all_events_view temporarily into temp_single_patient table
	DROP TABLE IF EXISTS temp_single_patient;				
    CREATE TABLE temp_single_patient AS
    SELECT * FROM get_all_events_view(input_icustay_id, selected_itemids_list);

	str := ' "charttime" timestamp without time zone, ';			-- text = field type , not icustay_id in here, because it would group all rows
   	FOR rec IN SELECT DISTINCT label								-- looping to get column heading string
        FROM temp_single_patient									
		ORDER BY temp_single_patient.label
	LOOP
    	str :=  str || '"' || rec.label || '" text' ||',';			-- str is used for defintion of table below
    END LOOP;
    str:= substring(str, 0, length(str));							-- remove last comma
	raise notice 'str: %', str;

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
	DELETE FROM temp_transposed_patient WHERE temp_transposed_patient.charttime = '0001-01-01 00:00:01';	-- remove DUMMY values
	
	SELECT * FROM patient_cohort_filtered WHERE patient_cohort_filtered.icustay_id = input_icustay_id INTO v_patient_record;
	ALTER TABLE temp_transposed_patient	
		ADD COLUMN hadm_id						integer, 
		ADD COLUMN icustay_id					integer, 
		ADD COLUMN intime						timestamp without time zone, 
		ADD COLUMN outtime						timestamp without time zone, 
		ADD COLUMN los 							double precision,
		ADD COLUMN icustays_count				bigint,
		ADD COLUMN age							numeric,
		ADD COLUMN patientweight				double precision,
		ADD COLUMN gender						varchar(5),
		ADD COLUMN ethnicity					varchar(200),
		ADD COLUMN first_service				varchar(20),
		ADD COLUMN dob 							date,
		ADD COLUMN dod 							date,
		ADD COLUMN death_in_hosp 				int,
		ADD COLUMN death_3_days 				int,
		ADD COLUMN death_30_days 				int,
		ADD COLUMN death_365_days 				int,
		ADD COLUMN subject_id					int,
		ADD COLUMN icd9_code					varchar(10),
		ADD COLUMN all_icd9_codes 				varchar[];											
	UPDATE temp_transposed_patient 
	SET 
		hadm_id = v_patient_record.hadm_id,
		icustay_id = input_icustay_id,
		intime = v_patient_record.intime,
		outtime = v_patient_record.outtime,
		los = v_patient_record.los,
		icustays_count = v_patient_record.icustays_count,
		age = v_patient_record.age,
		patientweight = v_patient_record.patientweight,
		gender = v_patient_record.gender,
		ethnicity = v_patient_record.ethnicity,
		first_service = v_patient_record.first_service,
		dob = v_patient_record.dob,	
		dod = v_patient_record.dod,						
		death_in_hosp = v_patient_record.death_in_hosp, 				
		death_3_days = v_patient_record.death_3_days, 				
		death_30_days = v_patient_record.death_30_days, 				
		death_365_days = v_patient_record.death_365_days, 				
		subject_id = v_patient_record.subject_id,
		icd9_code = v_patient_record.icd9_code,
		all_icd9_codes = v_patient_record.all_icd9_codes;

end; $body$
