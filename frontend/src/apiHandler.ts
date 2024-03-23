import awaitable from "./await_util";

const BASE_URL = 'http://localhost:5000';

export function getSimilarity(lang: string, text_1: string, text_2: string) {
    const endpoint = `${BASE_URL}/api/${lang}`;
    const fetch_conf = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text_1, text_2 }),
    };

    return new Promise((resolve, reject) => {
        (async () => {
            const [res, err] = await awaitable(fetch(endpoint, fetch_conf));
            if (err) return reject(err);

            const request_id = (await res.json()).id;

            const interval = 2000;
            const timeout = 1.25 * 60 * 1000;

            let waited_for = 0;
            const timer_id = setInterval(async () => {
                waited_for += interval;

                const endpoint_get = `${endpoint}/${request_id}`; 
                const [res, err] = await awaitable(fetch(`${endpoint_get}`));
                if (err) return alert(err);

                const data = (await res.json()).result;
                if (data !== "pending") {
                    clearInterval(timer_id);
                    resolve(data);
                }

                if (waited_for > timeout) {
                    clearInterval(timer_id);
                    reject("Request timed out");
                }
            }, interval);
        })()
    })    
}