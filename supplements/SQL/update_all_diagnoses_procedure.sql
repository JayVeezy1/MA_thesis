create or replace procedure public.update_diagnoses_all_icd9()		

language plpgsql
as $body$
declare
	temp_all_icd9_codes varchar[];
	admission_entry RECORD;
	icd9_code_entry RECORD;
	v_counter int;
	temp_stroke_type varchar(15);
	temp_infarct_type varchar(15);
	
begin
	-- Task: stroke_type and infarct_type simply does not belong into here. Just get all the icd9_codes in one list, get the type in other function or even in Python.


	-- 2) loop through each admission and then each available icd9_code
	-- idea: for every first admission (seq_num = 1) loop through every distinct icd9_code, append to temp_list, then add to new column 'all_icd9_codes'
	v_counter = '1';
	FOR admission_entry IN (select diagnoses_icd.hadm_id from mimiciii.diagnoses_icd 
							where diagnoses_icd.seq_num = '1'
							and diagnoses_icd.hadm_id IN (select distinct hadm_id from public.diagnoses_all_icd9 where stroke_type is null)		-- set hadm_id filters here!
							order by diagnoses_icd.hadm_id)	
		LOOP  
			temp_all_icd9_codes:= '{}';  		
			temp_stroke_type:= 'no_stroke';
			temp_infarct_type:= 'no_infarct';
			Raise Notice 'Current round is: %', v_counter;
			Raise Notice 'For hadm_id: %', admission_entry.hadm_id;
			v_counter = v_counter + '1';

			FOR icd9_code_entry IN (select distinct diagnoses_icd.icd9_code from mimiciii.diagnoses_icd where diagnoses_icd.hadm_id = admission_entry.hadm_id)
				LOOP
					-- set all_icd9_codes list
					temp_all_icd9_codes:= array_append(temp_all_icd9_codes, icd9_code_entry.icd9_code); 
					-- check stroke_type
					if temp_stroke_type = 'no_stroke' then
						if icd9_code_entry.icd9_code = '430' or icd9_code_entry.icd9_code = '431' 
						or icd9_code_entry.icd9_code = '432' or icd9_code_entry.icd9_code = '4329' then 
							temp_stroke_type = 'hemorrhagic';
						end if;

						if icd9_code_entry.icd9_code = '433' or icd9_code_entry.icd9_code = '4330' or icd9_code_entry.icd9_code = '4331' or icd9_code_entry.icd9_code = '4332' 
						or icd9_code_entry.icd9_code = '434' or icd9_code_entry.icd9_code = '4340' or icd9_code_entry.icd9_code = '43400' 
						or icd9_code_entry.icd9_code = '43401' or icd9_code_entry.icd9_code = '4341' or icd9_code_entry.icd9_code = '43411' 
						or icd9_code_entry.icd9_code = '435' or icd9_code_entry.icd9_code = '4350' or icd9_code_entry.icd9_code = '4351'
						or icd9_code_entry.icd9_code = '4353' or icd9_code_entry.icd9_code = '4359' or icd9_code_entry.icd9_code = '436' then 
							temp_stroke_type = 'ischemic';
						end if;

						if icd9_code_entry.icd9_code = '437' or icd9_code_entry.icd9_code = '4370' or icd9_code_entry.icd9_code = '4371' or icd9_code_entry.icd9_code = '4372' 
						or icd9_code_entry.icd9_code = '4373' or icd9_code_entry.icd9_code = '4374' or icd9_code_entry.icd9_code = '43400' 
						or icd9_code_entry.icd9_code = '4381' or icd9_code_entry.icd9_code = '43811' or icd9_code_entry.icd9_code = '4382' 
						or icd9_code_entry.icd9_code = '43820' or icd9_code_entry.icd9_code = '4383' or icd9_code_entry.icd9_code = '4384'
						or icd9_code_entry.icd9_code = '4385' or icd9_code_entry.icd9_code = '4388' or icd9_code_entry.icd9_code = '43882' or icd9_code_entry.icd9_code = '43885' then
							temp_stroke_type = 'other_stroke';
						end if;			
					end if;
					-- check infarct_type
					if temp_infarct_type = 'no_infarct' then
						if icd9_code_entry.icd9_code = '41070' or icd9_code_entry.icd9_code = '41071' or icd9_code_entry.icd9_code = '41072' then 
							temp_infarct_type = 'NSTEMI';
						end if;

						if icd9_code_entry.icd9_code = '41000' or icd9_code_entry.icd9_code = '41001' or icd9_code_entry.icd9_code = '41002' or icd9_code_entry.icd9_code = '41010' 
						or icd9_code_entry.icd9_code = '41011' or icd9_code_entry.icd9_code = '41012' or icd9_code_entry.icd9_code = '41020' 
						or icd9_code_entry.icd9_code = '41021' or icd9_code_entry.icd9_code = '41022' or icd9_code_entry.icd9_code = '41030' 
						or icd9_code_entry.icd9_code = '41031' or icd9_code_entry.icd9_code = '41032' or icd9_code_entry.icd9_code = '41040'
						or icd9_code_entry.icd9_code = '41041' or icd9_code_entry.icd9_code = '41042' or icd9_code_entry.icd9_code = '41050'
						or icd9_code_entry.icd9_code = '41051' or icd9_code_entry.icd9_code = '41052' or icd9_code_entry.icd9_code = '41060'
						or icd9_code_entry.icd9_code = '41061' or icd9_code_entry.icd9_code = '41062' or icd9_code_entry.icd9_code = '41080'
						or icd9_code_entry.icd9_code = '41081' or icd9_code_entry.icd9_code = '41082' or icd9_code_entry.icd9_code = '41090'
						or icd9_code_entry.icd9_code = '41091' or icd9_code_entry.icd9_code = '41092' or icd9_code_entry.icd9_code = '4110' or icd9_code_entry.icd9_code = '4111' then 
							temp_infarct_type = 'STEMI';
						end if;						
					end if;

				END LOOP;

			UPDATE public.diagnoses_all_icd9
			SET all_icd9_codes = temp_all_icd9_codes
			WHERE diagnoses_all_icd9.hadm_id = admission_entry.hadm_id;	
			UPDATE public.diagnoses_all_icd9
			SET stroke_type = temp_stroke_type
			WHERE diagnoses_all_icd9.hadm_id = admission_entry.hadm_id;	
			UPDATE public.diagnoses_all_icd9
			SET infarct_type = temp_infarct_type
			WHERE diagnoses_all_icd9.hadm_id = admission_entry.hadm_id;	

		END LOOP;         
	
end; $body$
