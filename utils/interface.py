import sys
from db.mdb import list_last_jobs

def find_old_job():
	# Interface for user to select an old job to fork
    last_jobs = list_last_jobs()
    if len(last_jobs) == 0:
        print "There's no previous job. Start over using 'train' mode."
        sys.exit(1)
    select = raw_input("Please enter id of the job: ")
    is_id_valid = False
    while not is_id_valid:
        try:
            select = int(select)
            selected_job = last_jobs[select-1]
            is_id_valid = True
        except ValueError:
            print "Entered value must be integer"
        except IndexError:
            print "There's no such job"
        finally:
        	if not is_id_valid:
        		select = raw_input("Please enter id of the job: ")
    return selected_job
