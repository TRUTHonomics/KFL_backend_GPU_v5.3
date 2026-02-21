# DB-tunnel voor GPU container (15432)

De container praat met `kflhyper` via `host.docker.internal:15432`. Die poort moet op de **Windows-host** door een SSH-tunnel worden bediend (VM120 = 10.10.10.3).

## Waarom de fout bleef

- Standaard `ssh -L 15432:...` luistert alleen op **127.0.0.1**. Docker gebruikt het **host-IP**; daar reageert niemand op → Connection refused.
- Oplossing: tunnel met **0.0.0.0:15432** starten (zie `start_db_tunnel.ps1`). Als je SSH handmatig in WSL/Git Bash start, luistert de tunnel in de Linux-VM; `host.docker.internal` wijst naar Windows → tunnel moet op Windows draaien.

## Handmatig (nu)

```powershell
.\scripts\start_db_tunnel.ps1
```
Venster open laten; tunnel stopt als je het sluit.

## Permanent (eenmalig)

```powershell
.\scripts\install_db_tunnel_task.ps1
```
Registreert een taak die bij **aanmelding** de tunnel start (verborgen). Geen admin nodig.

Daarna: container herstarten of gewoon opnieuw gebruiken; DB-init zou moeten slagen.
