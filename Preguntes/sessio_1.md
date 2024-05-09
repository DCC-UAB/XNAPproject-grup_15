# Preguntes fetes per la sessió 1:

Durant aquesta primera sessió, ens volem centrar en analitzar les dades que tenim. Això ens ajudarà a definir el nostre enfocament per a les pròximes sessions i elaborar un model que s'adapti al nostre problema. En el nostre cas, disposem de tres fonts de dades diferents, cadascuna amb el seu propi conjunt d'entrenament, de test i de validació. Aquestes fonts de dades són AFAD, CACD i MORPH. 

**AFAD (Asian Face Age Dataset):** És un conjunt de dades que conté imatges facials d'individus asiàtics classificades per edat. Sovint s'utilitza per a tasques de predicció d'edat a partir d'imatges facials.

**CACD (Cross-Age Celebrity Dataset):** És un conjunt de dades que conté imatges facials de celebritats en diferents edats. És útil per a tasques de reconeixement facial i predicció d'edat, especialment per la seva diversitat i variabilitat temporal.

**MORPH:** És un conjunt de dades que també conté imatges facials de persones en diverses etapes de la seva vida. Sol usar-se per a estudis de predicció d'edat i anàlisi del canvi facial al llarg del temps.

A partir d'aquestes fonts, volem respondre les següents preguntes:


## 1. Distribució d'Edats:

**Objectiu:** Assegurar que el model tingui prou dades de cada rang d'edat per aprendre adequadament i no esbiaixar-se cap a certs grups.

· Quin és el rang d'edats en cada conjunt (train, test i validation) per a cadascuna de les fonts de dades (AFAD, CACD, MORPH)?
· Està equilibrada la distribució d'edats en cada conjunt? Si no, hi ha grups d'edat amb una major representació que d'altres?

## 2. Quantitat de Dades:

**Objectiu:** Comprovar que hi ha prou dades per entrenar, validar i testejar el model, evitant el sobreajustament i l'infraajustament.

· Quin és el nombre d'imatges en cada conjunt (train, test, validation) per a cada font de dades?

· El conjunt d'entrenament (train) és prou gran per evitar el sobreajustament, és a dir, que el model no aprengui només les característiques específiques de les dades d'entrenament?

· El conjunt de validació i test tenen suficients dades per detectar el sobreajustament i evitar que el model tingui un rendiment baix en dades noves (underfitting)?

## 3. Biaixos Potencials:

**Objectiu:** Identificar possibles biaixos en els conjunts de dades que podrien afectar la precisió i l'objectivitat del model.

· Hi ha algun biaix evident en les dades (per exemple, més dades d'un grup d'edat específic, predominança d'un gènere)?

· Com podrien aquests biaixos afectar el rendiment del model a l'hora de predir l'edat en dades noves?

## 4. Soroll en les Imatges:

**Objectiu:** Detectar fonts de soroll que puguin afectar el rendiment del model i portar a sobreajustament o infraajustament.

· Les imatges presenten soroll visual, com ara marques d'aigua, ombres, il·luminació inconsistent o fons desordenats?

· En cas que hi hagi soroll, com es podria afectar el rendiment del model? Podria conduir a sobreajustament si el model aprèn a detectar elements irrellevants? O podria conduir a underfitting si el soroll dificulta l'aprenentatge de les característiques clau?













[18:44:15.163] Log Level: 2
[18:44:15.171] SSH Resolver called for "ssh-remote+lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com", attempt 1
[18:44:15.172] "remote.SSH.useLocalServer": false
[18:44:15.172] "remote.SSH.useExecServer": true
[18:44:15.172] "remote.SSH.showLoginTerminal": false
[18:44:15.173] "remote.SSH.remotePlatform": {"lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com":"linux"}
[18:44:15.173] "remote.SSH.path": undefined
[18:44:15.173] "remote.SSH.configFile": undefined
[18:44:15.173] "remote.SSH.useFlock": true
[18:44:15.173] "remote.SSH.lockfilesInTmp": false
[18:44:15.173] "remote.SSH.localServerDownload": auto
[18:44:15.173] "remote.SSH.remoteServerListenOnSocket": false
[18:44:15.177] "remote.SSH.showLoginTerminal": false
[18:44:15.178] "remote.SSH.defaultExtensions": []
[18:44:15.178] "remote.SSH.loglevel": 2
[18:44:15.178] "remote.SSH.enableDynamicForwarding": true
[18:44:15.178] "remote.SSH.enableRemoteCommand": false
[18:44:15.178] "remote.SSH.serverPickPortsFromRange": {}
[18:44:15.178] "remote.SSH.serverInstallPath": {}
[18:44:15.182] VS Code version: 1.89.0
[18:44:15.182] Remote-SSH version: remote-ssh@0.110.1
[18:44:15.182] win32 x64
[18:44:15.190] SSH Resolver called for host: lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com
[18:44:15.191] Setting up SSH remote "lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com"
[18:44:15.195] Using commit id "b58957e67ee1e712cebf466b995adf4c5307b2bd" and quality "stable" for server
[18:44:15.198] Install and start server if needed
[18:44:15.200] Checking ssh with "C:\Program Files (x86)\Common Files\Oracle\Java\javapath\ssh.exe -V"
[18:44:15.204] Got error from ssh: spawn C:\Program Files (x86)\Common Files\Oracle\Java\javapath\ssh.exe ENOENT
[18:44:15.204] Checking ssh with "C:\WINDOWS\system32\ssh.exe -V"
[18:44:15.205] Got error from ssh: spawn C:\WINDOWS\system32\ssh.exe ENOENT
[18:44:15.205] Checking ssh with "C:\WINDOWS\ssh.exe -V"
[18:44:15.206] Got error from ssh: spawn C:\WINDOWS\ssh.exe ENOENT
[18:44:15.206] Checking ssh with "C:\WINDOWS\System32\Wbem\ssh.exe -V"
[18:44:15.207] Got error from ssh: spawn C:\WINDOWS\System32\Wbem\ssh.exe ENOENT
[18:44:15.207] Checking ssh with "C:\WINDOWS\System32\WindowsPowerShell\v1.0\ssh.exe -V"
[18:44:15.208] Got error from ssh: spawn C:\WINDOWS\System32\WindowsPowerShell\v1.0\ssh.exe ENOENT
[18:44:15.209] Checking ssh with "C:\WINDOWS\System32\OpenSSH\ssh.exe -V"
[18:44:15.266] > OpenSSH_for_Windows_8.1p1, LibreSSL 3.0.2

[18:44:15.271] Running script with connection command: "C:\WINDOWS\System32\OpenSSH\ssh.exe" -T -D 59458 "lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com" bash
[18:44:15.274] Terminal shell path: C:\WINDOWS\System32\cmd.exe
[18:44:15.735] > ]0;C:\WINDOWS\System32\cmd.exe
[18:44:15.736] Got some output, clearing connection timeout
[18:44:16.030] > @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
[18:44:16.054] > @    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
> IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
> Someone could be eavesdropping on you right now (man-in-the-middle attack)!     
> It is also possible that a host key has just been changed.
> The fingerprint for the ECDSA key sent by the remote host is
> SHA256:AWOsaAfaoQOLpVxT7ROCyU6DdxaKE2Cg9sjFjfMNlhM.
> Please contact your system administrator.
> Add correct host key in C:\\Users\\eduar/.ssh/known_hosts to get rid of this mes
> sage.
> Offending ECDSA key in C:\\Users\\eduar/.ssh/known_hosts:1
> ECDSA host key for [lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp
> .azure.com]:5012 has changed and you have requested strict checking.
> Host key verification failed.
> El proceso ha intentado escribir en una canalización que no existe.
> 
[18:44:17.306] "install" terminal command done
[18:44:17.306] Install terminal quit with output: El proceso ha intentado escribir en una canalización que no existe.
[18:44:17.307] Received install output: El proceso ha intentado escribir en una canalización que no existe.
[18:44:17.307] Failed to parse remote port from server output
[18:44:17.310] Resolver error: Error: 
	at g.Create (c:\Users\eduar\.vscode\extensions\ms-vscode-remote.remote-ssh-0.110.1\out\extension.js:2:499181)
	at t.handleInstallOutput (c:\Users\eduar\.vscode\extensions\ms-vscode-remote.remote-ssh-0.110.1\out\extension.js:2:496503)
	at t.tryInstall (c:\Users\eduar\.vscode\extensions\ms-vscode-remote.remote-ssh-0.110.1\out\extension.js:2:620043)
	at async c:\Users\eduar\.vscode\extensions\ms-vscode-remote.remote-ssh-0.110.1\out\extension.js:2:579901
	at async t.withShowDetailsEvent (c:\Users\eduar\.vscode\extensions\ms-vscode-remote.remote-ssh-0.110.1\out\extension.js:2:583207)
	at async k (c:\Users\eduar\.vscode\extensions\ms-vscode-remote.remote-ssh-0.110.1\out\extension.js:2:576866)
	at async t.resolve (c:\Users\eduar\.vscode\extensions\ms-vscode-remote.remote-ssh-0.110.1\out\extension.js:2:580578)
	at async c:\Users\eduar\.vscode\extensions\ms-vscode-remote.remote-ssh-0.110.1\out\extension.js:2:846687
[18:44:17.324] ------




[18:44:18.322] Opening exec server for ssh-remote+lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com
[18:44:18.342] Initizing new exec server for ssh-remote+lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com
[18:44:18.344] Using commit id "b58957e67ee1e712cebf466b995adf4c5307b2bd" and quality "stable" for server
[18:44:18.346] Install and start server if needed
[18:44:18.365] Opening exec server for ssh-remote+lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com
[18:44:18.369] Running script with connection command: "C:\WINDOWS\System32\OpenSSH\ssh.exe" -T -D 59458 "lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com" bash
[18:44:18.370] Terminal shell path: C:\WINDOWS\System32\cmd.exe
[18:44:18.814] > ]0;C:\WINDOWS\System32\cmd.exe
[18:44:18.817] Got some output, clearing connection timeout
[18:44:19.033] > @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
[18:44:19.063] > @    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
> IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
> Someone could be eavesdropping on you right now (man-in-the-middle attack)!     
> It is also possible that a host key has just been changed.
> The fingerprint for the ECDSA key sent by the remote host is
> SHA256:AWOsaAfaoQOLpVxT7ROCyU6DdxaKE2Cg9sjFjfMNlhM.
> Please contact your system administrator.
> Add correct host key in C:\\Users\\eduar/.ssh/known_hosts to get rid of this mes
> sage.
> Offending ECDSA key in C:\\Users\\eduar/.ssh/known_hosts:1
> ECDSA host key for [lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp
> .azure.com]:5012 has changed and you have requested strict checking.
> Host key verification failed.
> El proceso ha intentado escribir en una canalización que no existe.
> 
[18:44:20.323] "install" terminal command done
[18:44:20.324] Install terminal quit with output: El proceso ha intentado escribir en una canalización que no existe.
[18:44:20.324] Received install output: El proceso ha intentado escribir en una canalización que no existe.
[18:44:20.325] Failed to parse remote port from server output
[18:44:20.326] Exec server for ssh-remote+lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com failed: Error
[18:44:20.326] Existing exec server for ssh-remote+lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com errored (Error)
[18:44:20.327] Initizing new exec server for ssh-remote+lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com
[18:44:20.327] Using commit id "b58957e67ee1e712cebf466b995adf4c5307b2bd" and quality "stable" for server
[18:44:20.330] Error opening exec server for ssh-remote+lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com: Error
[18:44:20.330] Install and start server if needed
[18:44:20.337] Running script with connection command: "C:\WINDOWS\System32\OpenSSH\ssh.exe" -T -D 59458 "lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com" bash
[18:44:20.339] Terminal shell path: C:\WINDOWS\System32\cmd.exe
[18:44:20.722] > ]0;C:\WINDOWS\System32\cmd.exe
[18:44:20.722] Got some output, clearing connection timeout
[18:44:20.929] > @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
[18:44:20.945] > @    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
> IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
> Someone could be eavesdropping on you right now (man-in-the-middle attack)!     
> It is also possible that a host key has just been changed.
> The fingerprint for the ECDSA key sent by the remote host is
> SHA256:AWOsaAfaoQOLpVxT7ROCyU6DdxaKE2Cg9sjFjfMNlhM.
> Please contact your system administrator.
> Add correct host key in C:\\Users\\eduar/.ssh/known_hosts to get rid of this mes
> sage.
> Offending ECDSA key in C:\\Users\\eduar/.ssh/known_hosts:1
> ECDSA host key for [lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp
> .azure.com]:5012 has changed and you have requested strict checking.
> Host key verification failed.
> El proceso ha intentado escribir en una canalización que no existe.
> 
[18:44:22.207] "install" terminal command done
[18:44:22.208] Install terminal quit with output: El proceso ha intentado escribir en una canalización que no existe.
[18:44:22.208] Received install output: El proceso ha intentado escribir en una canalización que no existe.
[18:44:22.208] Failed to parse remote port from server output
[18:44:22.209] Exec server for ssh-remote+lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com failed: Error
[18:44:22.209] Error opening exec server for ssh-remote+lab-799f3fca-34a7-4eb4-94da-faa144c7dfac.westeurope.cloudapp.azure.com: Error
