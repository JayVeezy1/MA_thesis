create or replace function get_all_diagnoses()		
returns table (															
		row_id						int,
		subject_id					int,
		hadm_id						int,
		seq_num						int,
		icd9_code					varchar(10),
		all_icd9_codes 				varchar[]			-- this column will be newly created, it contains all accessible diagnoses icd9 codes
) 
language plpgsql
as $body$
declare
	temp_all_icd9_codes varchar[];
	admission_entry RECORD;
	icd9_code_entry RECORD;
	v_counter int;
	
begin
	if (SELECT EXISTS (
			SELECT FROM pg_tables
			WHERE schemaname = 'public' AND tablename  = 'diagnoses_all_icd9')) 
	then	
		return query select * from public.diagnoses_all_icd9;
	else
		-- 1) create the new table
		DROP TABLE diagnoses_all_icd9;  -- -> once this is created for all entries, do not delete table again!
		CREATE TABLE diagnoses_all_icd9 (
				row_id						int,
				subject_id					int,
				hadm_id						int,
				seq_num						int,
				icd9_code					varchar(10),
				all_icd9_codes 				varchar[] DEFAULT '{none}'
		);
		INSERT INTO diagnoses_all_icd9 (row_id, subject_id, hadm_id, seq_num, icd9_code)
		SELECT 
			diagnoses_icd.row_id, 
			diagnoses_icd.subject_id, 
			diagnoses_icd.hadm_id, 
			diagnoses_icd.seq_num, 
			diagnoses_icd.icd9_code 
		FROM mimiciii.diagnoses_icd;

		-- 2) loop through each admission and then each available icd9_code
		-- idea: for every first admission (seq_num = 1) loop through every distinct icd9_code, append to temp_list, then add to new column 'all_icd9_codes'
		v_counter = '1';
		FOR admission_entry IN (select diagnoses_icd.hadm_id from mimiciii.diagnoses_icd 
								where diagnoses_icd.seq_num = '1'
								order by diagnoses_icd.hadm_id)	
			LOOP  
				temp_all_icd9_codes:= '{}';  		-- hatte ich den vergessen??
				-- Raise Notice 'Current round is: %', v_counter;
				-- Raise Notice 'For hadm_id: %', admission_entry.hadm_id;
				v_counter = v_counter + '1';

				FOR icd9_code_entry IN (select distinct diagnoses_icd.icd9_code from mimiciii.diagnoses_icd where diagnoses_icd.hadm_id = admission_entry.hadm_id)
					LOOP
						temp_all_icd9_codes:= array_append(temp_all_icd9_codes, icd9_code_entry.icd9_code);   
					END LOOP;

				UPDATE diagnoses_all_icd9
				SET all_icd9_codes = temp_all_icd9_codes
				WHERE diagnoses_all_icd9.hadm_id = admission_entry.hadm_id;	

			END LOOP;         
		RETURN Query (SELECT * FROM diagnoses_all_icd9);
		
	END IF;
	
end; $body$
