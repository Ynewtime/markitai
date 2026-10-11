# Cloud credential fixtures

`test-rsa-pkcs8.b64` is a 2048-bit RSA key generated for these tests only
(`openssl genpkey`, 2026-10-11) and never registered anywhere: the base64 of
its PKCS#8 DER, stored without PEM armor; the tests wrap it into the
`private_key` of a synthetic Google service-account file. `test-rsa-public.b64`
is its public key (PKCS#1 `RSAPublicKey` DER, base64), used to verify the
signed token requests.
