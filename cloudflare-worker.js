export default {
  async fetch(request, env) {
    const cors = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
      "Access-Control-Allow-Headers": "*",
      "Access-Control-Max-Age": "86400"
    };
    if (request.method === "OPTIONS") return new Response(null,{status:204,headers:cors});
    const incoming=new URL(request.url);
    const base=String(env.RAILWAY_ORIGIN||"https://tacticai-worker-production.up.railway.app").replace(/\/$/,"");
    const path=incoming.pathname.replace(/^\/api/,"")||"/";
    const target=base+path+incoming.search;
    try {
      const headers=new Headers(request.headers);
      headers.delete("host"); headers.delete("content-length");
      const init={method:request.method,headers,redirect:"follow"};
      if(request.method!=="GET"&&request.method!=="HEAD") init.body=request.body;
      const upstream=await fetch(target,init);
      const out=new Response(upstream.body,upstream);
      for(const [k,v] of Object.entries(cors)) out.headers.set(k,v);
      out.headers.set("Cache-Control","no-store");
      return out;
    } catch(error) {
      return new Response(JSON.stringify({error:"cloudflare_upstream_error",message:"Cloudflare no pudo conectar con Railway",target,detail:String(error&&error.message||error)}),{status:502,headers:{...cors,"Content-Type":"application/json","Cache-Control":"no-store"}});
    }
  }
};