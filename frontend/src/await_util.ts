export default async function awaitable(promise: Promise<any>) {
    return promise.then(data => [data, null]).catch(error => [null, error]);
}