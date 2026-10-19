"""Canonical entry point, with additional response-identity audit only."""
import asyncio
import fcntl
import json
from common import *
import driver

ORIGINAL_QUERY=driver.ORIGINAL_QUERY
ORIGINAL_GRADE=driver.grade
ORIGINAL_DUMP=driver.dump
ORIGINAL_LOG=driver.log
ORIGINAL_VALID=driver.valid_result

class EvidencePersistenceFailure(BaseException):
    """Stop the process; an I/O problem must never buy another judgment draw."""

def durable_dump(*args,**kwargs):
    try:
        return ORIGINAL_DUMP(*args,**kwargs)
    except Exception as error:
        raise EvidencePersistenceFailure('Cannot persist evidence; resume from first valid pending result') from error

def durable_log(*args,**kwargs):
    try:
        return ORIGINAL_LOG(*args,**kwargs)
    except Exception as error:
        raise EvidencePersistenceFailure('Cannot persist attempt log; stop before another paid draw') from error

def readable_result(*args,**kwargs):
    try:
        return ORIGINAL_VALID(*args,**kwargs)
    except OSError as error:
        raise EvidencePersistenceFailure('Cannot read accepted evidence; stop before another paid draw') from error

async def identified_query(**kwargs):
    async for message in ORIGINAL_QUERY(**kwargs):
        fields={field:getattr(message,field) for field in ['message_id','uuid','session_id','model','stop_reason','terminal_reason','api_error_status'] if getattr(message,field,None) is not None}
        if fields:
            with (driver.CURRENT.get()/'response-identities.jsonl').open('a') as out:
                out.write(json.dumps({'type':type(message).__name__,**fields})+'\n')
        yield message

async def first_valid_after_restart(case,phase,cooldown,semaphore):
    folder=HERE/'judging'/phase/case['job_id']
    target=folder/'result.json'
    if not target.exists():
        # A crash between validating an attempt and atomically publishing its
        # result must not cause a second paid draw to replace that valid grade.
        for attempt in sorted(folder.glob('attempt-*'),key=lambda p:int(p.name.split('-')[1])):
            pending=attempt/'candidate-result.json'
            if pending.exists():
                try:
                    valid=driver.valid_result(pending,case)
                except (ValueError,KeyError,AssertionError):
                    valid=False
                if valid:
                    dump(target,read(pending))
                    driver.log('recovered_first_valid',phase=phase,job_id=case['job_id'],attempt=attempt.name)
                    break
    await ORIGINAL_GRADE(case,phase,cooldown,semaphore)

def main():
    from acceptance import verify_acceptance
    verify_acceptance(create=True)
    assert read(HERE/'VALIDATION.json')['passed']
    registration={'driver_freeze_sha256':sha(HERE/'driver-sha256.json'),
                  'canonical_entry_sha256':sha(HERE/'canonical.py'),
                  'validation_sha256':sha(HERE/'VALIDATION.json'),
                  'validation_acceptance_sha256':sha(HERE/'VALIDATION-ACCEPTED.json'),
                  'ownership':'All changed cases receive separately designated first structurally valid canonical request; diagnostics cannot become canonical.',
                  'audit_addition':'Capture SDK message_id/uuid; recover the first validated pending result after restart; fail closed on evidence/log I/O errors instead of buying another draw. Unchanged request options/text/acceptance and eight-request scheduling.'}
    path=HERE/'canonical-registration.json'
    if path.exists():assert read(path)==registration
    else:dump(path,registration)
    driver.ORIGINAL_QUERY=identified_query
    driver.grade=first_valid_after_restart
    driver.dump=durable_dump
    driver.log=durable_log
    driver.valid_result=readable_result
    driver._CANONICAL_WRAPPER_ACTIVE=True
    with (HERE/'.judge.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(driver.main('canonical'))

if __name__=='__main__':main()
