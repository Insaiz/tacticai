export default {
  async fetch(request, env) {
    const target = new URL(request.url);
    const base = String(env.RAILWAY_ORIGIN || 'https://tacticai-worker-production.up.railway.app').replace(/\/$/, '');
    const path = target.pathname.replace(/^\/api/, '') || '/';
    const url = base + path + target.search;
    const headers = new Headers(request.headers);
    headers.delete('host');
    headers.delete('content-length');
    const init = { method: request.method, headers, redirect: 'follow' };
    if (request.method !== 'GET' && request.method !== 'HEAD') init.body = request.body;
    const response = await fetch(url, init);
    const out = new Response(response.body, response);
    out.headers.set('Access-Control-Allow-Origin', '*');
    out.headers.set('Access-Control-Allow-Methods', 'GET,POST,PUT,OPTIONS');
    out.headers.set('Access-Control-Allow-Headers', '*');
    out.headers.set('Cache-Control', 'no-store');
    return out;
  }
};
