"""Record structural evidence of native state continuity, without reasoning text."""
import hashlib,json
from multi_turn_eval.services.openai_responses import OpenAIResponsesLLMService

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def summary(item):
 return {**{k:item[k] for k in ['type','role','id','call_id','name','phase','status'] if k in item},
         'sha256':digest(item),'content_characters':sum(len(p.get('text','')) for p in item.get('content',[]) if isinstance(p,dict)),
         'has_encrypted_reasoning':bool(item.get('encrypted_content'))}
def install(path):
 cls=OpenAIResponsesLLMService;original_request=cls._responses_request_params;original_complete=cls._remember_completed_response
 sequence=0
 def write(record):
  with path.open('a') as out:out.write(json.dumps(record,sort_keys=True)+'\n')
 def request(self,context):
  nonlocal sequence
  params=original_request(self,context);sequence+=1
  write({'event':'request','sequence':sequence,'model':params['model'],'previous_response_id':params.get('previous_response_id'),
         'store':params.get('store'),'reasoning':params.get('reasoning'),'service_tier':params.get('service_tier'),
         'sampling':{k:params[k] for k in ['temperature','top_p','max_output_tokens'] if k in params},
         'input_sha256':digest(params['input']),'input_items':[summary(i) for i in params['input']],
         'native_history_sha256':digest(self._native_history)})
  return params
 def complete(self,response):
  original_complete(self,response)
  output=self._get_attr(response,'output') or []
  write({'event':'completed','sequence':sequence,'id':self._get_attr(response,'id'),
         'model':self._get_attr(response,'model'),'service_tier':self._get_attr(response,'service_tier'),
         'output_items':[summary(self._native_item(i)) for i in output],
         'native_history_sha256':digest(self._native_history)})
 cls._responses_request_params=request;cls._remember_completed_response=complete
