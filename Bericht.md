# Bericht
## Baseline
Als Baseline habe ich einen Random Agent genommen. Dieser sampled seine Actions uniform vom Action-Space. Das heisst, er lernt nichts und reagiert auch nicht auf das Environment oder dessen Observation.
Ich erwarte keinen grossen Reward, da die Actions zufällig gewählt werden. Allerdings erwarte ich auch nicht einen Reeward von 0, da man mit zufälligen Drücken von Tasten immernoch das ein oder andere Alien abschiessen kann. Vielmehr soll der Random Agent eine Grundlage geben, was man durch zufälliges Drücken von Tasten eerreichne kann um in den nächsten Experimenten einen Verlgeich zu haben.
![](assets/Pasted image 20241222150011.png]
Der Random Agent hat einen durchschnittlichen Return von 136.8 erreicht, wobei es einige Ausreisser nach oben gibt wie in der Abbbildung gut zu sehen ist.. Das erklärt auch der ein bisschen tiefere Median. 

Beim Ansehen der Videos wird gut ersichttlich, dass es sich hier um zufällige Actioons handelt. Das Spaceship bewegt sich schnell hin und her und legt keine längeren Strecken an einem St¨ück in eine bestimmte Richtung zurück.
## Initialer Ansatz
Als initialer Ansatz habe ich mich für Proximal Policy Optimization (PPO) entschieden. Das ist ein Policy-Based Reinforcement Learning Algorithmus. Des Weiteren ist er On-Policy, was so viel bedeutet wie, dass beim Explorrieren die gleiche Policy verwendet wird wie beim Aktualisieren der Policy. Genauer gesagt ist PPO ein Actor-Critic Algorithmus, welcher ein Policy-basierter Ansatz (Actor) mit einem Value-basierten Ansatz (Critic) über eine Advantage-Function kombiniert.

PPO habe ich aus mehreren Gründen ausgewählt:
1. Es hat mich interessiert, den RL-Ansatz welcher hinter ChatGPT ist besser kennenzulernen.
2. Gemäss diesem Paper ([https://arxiv.org/pdf/2306.01451](https://arxiv.org/pdf/2306.01451)) outperformed PPO Deep Q-Network (DQN)
3. Als ich die beiden zur Verfügung gestellten clean_rl.py-Scripts getestet habe, brauchte PPO wesentlich weniger Zeitschritte als DQN um zu lernen (Faktor 10). Und damit auch Faktor 10 mal weniger Compute-Time.
4. Gemäss [Wikipedia](https://en.wikipedia.org/wiki/Proximal_policy_optimization#Advantages) bringt PPO hauptsächlich drei Vorteile mit sich: 
	1. PPO ist einfach zu berechnen. Gerade verglichen mit TRPO, was PPO ja versucht zu approximieren.
	2. PPO ist stabil weil es nicht viel Hyperparameter-Tuning braucht.
	3. PPO ist Sample-Efficienct und lerrnt daher schneller. Dies ist generall der Fall bei On-Policy Methoden. Algorithmen wie DQN (welche Off-Policy sind) sind viel weniger Sample-Efficient.
### Hyperparameter
Hier sind Hyperparameter beschreiben, welche relevant für das Experiment sind. Parameter wie `wandb` werden nicht beschrieben, da sie keinen Einfluss auf das Experiment haben sondern in diesem Fall jetzt für das Loggen der Werte verwendet wurden.

| param           | value                | Beschreibung                                                                                                                                                                                                                                                        |
| --------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| env_id          | ALE/SpaceInvaders-v5 | Umgebung in der der Agent traiiniert.                                                                                                                                                                                                                               |
| total_timesteps | 1000000              | Die Gesamtanzahl der Schritte, in denen der Agent mit dem Environment interagiert.                                                                                                                                                                                  |
| learning_rate   | 0.00025              | Wie schnell/fest die Parameter des Models angepasst werden.                                                                                                                                                                                                         |
| num_envs        | 16                   | Wie viele Environments werden parallel simuliert?                                                                                                                                                                                                                   |
| num_steps       | 128                  | Wie viele Schritte in jedem Environment pro Policy Rollout simuliert werden.                                                                                                                                                                                        |
| anneal_lr       | True                 | Wenn `True`, wird die Lernrate während des Trainings linear verringert. Dies stabilisiert das Training, indem die Lernrate im Verlauf des Trainings reduziert wird.                                                                                                 |
| gamma           | 0.99                 | Der Discount-Factor für zukünftige Rewards. Balanciert die Wichtigkeit zwischen unmittelbaren und langfristigen Rewards. Werte nahe 1 bevorzugen langfristige Rewards.                                                                                              |
| gae_lambda      | 0.95                 | Lambda-Parameter für Generalized Advantage Estimation (GAE). Balanciert Bias und Varianz bei der Schätzung des Vorteils: Niedrigere Werte bevorzugen hohen Bias und niedrige Varianz, höhere Werte das Gegenteil.                                                   |
| num_minibatches | 4                    | Anzahl der Mini-Batches, in die die Rollout-Daten für die Gradienten-Updates aufgeteilt werden. Kleinere Mini-Batches ermöglichen mehr Updates pro Epoche, erhöhen aber die Varianz. Durch erhöhte Varianz können Modelle besser aus einem lokalen Minimum escapen. |
| update_epochs   | 4                    | Wie oft die gleichen Daten innerhalb eines Rollout zum updaten verwendet werden. Mehr Epochen verbessern das Lernen aus denselben Daten, können jedoch zu Overfitting führen.                                                                                       |
| norm_adv        | True                 | Normalisiert die Advantagefunktion. Hilft Updates zu stabilisieren.                                                                                                                                                                                                 |
| clip_coef       | 0.1                  | Der Clipping-Coefficient für die PPO-Optimierungsfunktion. Verhindert grosse Updates, indem Änderungen der Policy eingeschränkt werden, was die Stabilität verbessert.                                                                                              |
| clip_vloss      | True                 | Ob der Coefficient in der Loss-Funktion clipped werden soll oder nicht.                                                                                                                                                                                             |
| ent_coef        | 0.01                 | Fördert Exploration, indem Entropie zur Verlustfunktion hinzugefügt wird; höhere Werte führen zu stärker explorativem Verhalten.                                                                                                                                    |
| vf_coef         | 0.5                  | Actor-Critic: Balanciert die Bedeutung des Loss der Value-Function im Vergleich zum Policy-Loss.                                                                                                                                                                    |
| max_grad_norm   | 0.5                  | Maximal erlaubter Normwert des Gradienten beim Clipping.                                                                                                                                                                                                            |
| target_kl       | None                 | Beschränkung der KL-Divergenz zwischen den Updates, da die Beschränkung nicht ausreicht, um grosse Updates zu verhindern.                                                                                                                                           |
### Wrappers
Wrappers haben auch einen wesentlichen Einfluss auf das Experiment. Sie bestimmen vorallem Eigenschaften des Environments, respektive wie das Environment einen Einfluss auf den Agenten hat (ResizeObservation). 

| Wrapper                          | Funktion                                                                                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NoopResetEnv(env, noop_max=30)   | Tätigt bis zu 30 zufällige Actions beim REset eines Environment.                                                                                                   |
| MaxAndSkipEnv(env, skip=4)       | Frameskipping: Gibt nur das 4te Frame zurück und dabei das Maximum zwischen denn letzten beiden Frames.                                                            |
| ResizeObservation(env, (84, 84)) | Resized das Bild (die Observation) auf 84x84 Pixel. Dabei geht ein wenig Information verloren aber das Modell muss weniger Parameter haben/Berechnungen ausführen. |
| GrayscaleObservation(env)        | Macht das Bild Schwarz-Weiss (mit Graustufen).                                                                                                                     |
| FrameStackObservation(env, 4)    | Stacked die letzten 4 Frames zu einem.                                                                                                                             |
*Wie verhählt sich die Kombination von `MaxAndSkipEnv` und `FrameStackObservation`?* => Durch `MaxAndSkipEnv` und `FrameStackObservation` führt nur jedes 16te Frame zu eine Action vom Agent.
### Agent
Das verwendete Modell sieht wie folgt aus. Es besteht aus einem Encoder mit 3 Conv2D Layer und einem linearen Layer. Der Encoder erhält als Input 4 Grayscale Bilder mit einer Grösse von 84x84. Die 4 Bilder sind gestacked sodass es zu einem Bild mit 4 Channels wird. Der Encoder bläst die Channels im ersten Conv2D Layer durch einen 8x8 Kernel mit einer Stride von 4 auf 32 Channels auf. Dann mit einem 4x4 Kernel und einem Stride von 2 auf 64 Channels. Anschliessend folgt ein letzter Conv2D Layer mit einem 3x3 Layer und einem Stride von 1 aber die Anzahl Channels bleiben gleich. Dazwischen befindet sich jeweils eine ReLu-Aktivierungsfunktion. Das resultierende Bild mit einer Grösse von 7x7 Pixel und 64 Channels wird direkt in einen neuronalen Layer gefüttert und auf 512 Outputneuronen reduziert.
Der Actor und Critic haben jeweils einen separaten Head bestehend aus einem linearen Layer 512 Inputs und mit 6 Outputs (Actor) respektive 1 Output für den Critic.

Dieses Modell wird für jeden Agent verwendet, ausser im Baseline-Setup und im ResNet18-Setup.
### Vergleich mit Baseline
Die folgende Grafik zeigt die Verteilung der Returns von 100 Episoden in der Evaluationen dar (y-Achse). Zusehen ist hier ein Violinplot mit zwei Verteilungen. Eine des Baseline Ansatzes mit einem Random Agent und eine für den initialen Ansatz (PPO). In Rot ist jeweils der Median eingezeichnet und in Blau das 10te und 90ste Percentil. Alle ähnlichen Grafik sind gleich aufgebaut. Sie unterscheiden sich lediglich darin, welche Konfiguration(en) abgebildet ist/sind auf der x-Achse und ob die Verteilung der Returns der Episoden oder die Verteilungen der Dauer pro Epsiode oder Länge pro Episode dargestellt wird.
![](assets/Pasted image 20241230161019.png]
Wie auf ddem Plot zu sehen ist, ist der Median der Runs des initialen Ansatzes deutlich über dem 90% Percentil des Baseline (Random Agent). Es ist also klar zu sehen, dass der Agent mit PPO gelernt hat.


![](assets/Pasted image 20241230160944.png]
Auch bei den Anzahl Schritten pro Episode liegt der Median der Runs des initialen Ansatzes deutlich über dem 90% Percentil des Baseline-Ansatzes. Dies scheint logisch zu sein, da mehr Reward durch das Abschiessen von mehr Aliens erreicht wird und dies mehr Zeit beansprucht.

![](assets/Pasted image 20241230160956.png]
Bei der Episodendauer ist der Unterschied noch extremer. Erstens, weil mehr Zeitschritte länger dauern und zweitens, weil der initiale Ansatz noch viel mehr Berechnung für den Agenten ausführen muss um die nächste Action auszuwählen. Während im Baseline-Ansatz nur eine Zufallszahl gezogen wird.
## Mögliche Erweiterungen
Mein initialer Ansatz kann man auf diverse Arten erweitern. Ich habe hier die 4 Varianten aufgelistet welche ich umgesetzt habe.
1. **Hyperparameter Tuning:** Die Koeffiziente welche das Clipping begrenzt `clip_coef` ist relativ zentral für PPO. Diese werde ich in diesem Experiment tunen.
2. **Architektur des Agenten:** Viele Modelle welche mit Bilder Arbeiten oder Bilder erkenne arbeiten mit der Resnet-Architektur. Diese werde ich in diesem Experiment testen.
3. **Balance von Exploration vs. Exploitation verbessern:** Ich werde in diesem Experiment versuchen die Balance zwischen Exploration und Exploitation durch ein Intrinsic Exploration Module (IEM-PPO) aus diesem Paper zu verbessern: [https://arxiv.org/pdf/2011.05525](https://arxiv.org/pdf/2011.05525). Das Paper hat mit IEM-PPO bessere Returns erreicht sowie eine tiefere Varianz. 
4. **Frame Skip Adjustment:** In diesem Experiment werde ich herausfinden wie sich das Frame Skipping auf den Return auswirkt.
## 1. Hyperparameter Tuning
Ein zentrales Element von PPO ist die Clip-Funktion. Diese hat einen Parameter epsilon welche den Bereich um 1 angibt in welcher der Wert der Ratio-Function $r_t(\theta)$ clipped wird. Das clipping selbst sieht so aus: $$\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)$$
Der Parameter $\epsilon$ ist im initialen Ansatz auf $0.1$ gesetzt. Ich habe aber an diversen Orten gelesen, dass ein default Wert von $0.2$ gut ist. Ich werde deshalb den Hyperparameter $\epsilon$ mit den werden $\{0.1, 0.2, 0.3, 0.4\}$ testen. Ich erwarte, dass PPO mit $\epsilon=0.2$ gleich gut oder ein bisschen besser wird. Ab $\epsilon>0.2$ erwarte ich eine schlechtere Performance in der Evaluation aufgrund eines instabieleren Lernens während dem Training aufgrund der höheren Varianz durch den höheren Parameter Epsilon.
![](assets/Pasted image 20241225181208.png]
Es ist gut zu beobachten, dass das Model mit $\epsilon=0.2$ besser abgschneidet. Aber wenn $\epsilon$ grösser wird, dann verschlechtert sich der Return wieder. Der initiale Parameter $\epsilon$ scheint im Script von CleanRL nicht ein optimal gewählter Default-Wert zu sein und auch für diesen Anwendungsfall von PPO scheint der viel empfohlene Standardwert von $0.2$ die besseren Resultate zu liefern. 
## 2. Resnet-Architektur
Viele Bild-Klassifizierungstasks verwenden die ResNet Architektur und erzielen dabei sehr gute Resultate. Deshalb möchte ich wissen, wie sich ResNet mit PPO auf Space-Invaders verhaltet. Ich werde das Default-CNN durch ResNet austauschen. Die Linearen Layer welche spezifisch für den Actor und Critic sind, behalte ich bei. Im ersten Versuch übernehme ich  keine weights sondern trainiere ResNet von Grund auf neu. In einer zweiten Konfiguration habe ich dann noch mit vortrainierten weights gearbeitet von https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights. 
Ich vermute, dass das Modell mit ResNet18 und neu initialisierten Weights schlechter abschneiden wird. Da ResNet18 circa 10mal mehr Parameter hat und mehr Trainingsdaten brauchen wird um effektiv etwas zu lernen. Beim ResNet18-Modell mit vortrainierten Weights vermute ich, dass es besser abschneiden wird als die initiale Konfiguration, da es bereits Vorwissen über Bilder hat und dieses in anderen Anwendungsfällen wie Bilderkennung effektiv gebraucht werden konnte und das Modell so bessere Ergebnisse erziehlte als ohne vortrainierte Weights. Den gleicehn Effekt erhoffe ich mir auch hier.

![](assets/Pasted image 20241225181924.png]

Das ResNet18 Modell ohne vortrainierte Weights hat wie erwartet ein bisschen schlechter abgeschnitten als die initiale Konfiguration. Überraschend war aber, dass das Modell nicht von den vortrainierten Weights profitieren konnte und diese das Modell gehindert haben beim Lernen. Ich habe die Vermutung, dass es daran liegen könnte, dass sehr wenig Screenshots von Atari-Games oder allgemein Games im Bilderdatensatz sind, welcher verwendet wurde um das Modell Vorzutrainieren. Ein kurzer Blick in die Klassen zeigt, die Begriffe Games, Atari, Console, Screenshot, Alien, Retro sind überhaupt nicht zu finden. Nur folgende Begriffe die mit SpaceInvaders zusammenhängen habe ich gefunden: Space shuttle und CRT screen. Ich vermute, dass das Modell diese Art von Bilder noch nicht viel gesehen hat.

![](assets/Pasted image 20241225183049.png]
Auch interessant ist, dass das Modell deutlich länger pro Episode braucht, während sich die Episodenlänge nicht verlängert hat sondern auch noch gekürzt. Das lässt sich aber einfach auf das kompliziertere Modell zurückführen, welches viel mehr Parameter hat und dementsprechend auch viel mehr Berechnungen durchführen muss.
![](assets/Pasted image 20241225183205.png]
## 3. Intrinsic Exploration Module
Gemäss diesem Paper: https://arxiv.org/pdf/2011.05525 kann mit Intrinsic Exploration Module (IEM) bessere Performance bei kleinerer Varianz erzielt werden.  
Der IEM-PPO Algorithmus unterscheidet sich wie folgt (Ergänzungen sind in gelb markiert):
![](assets/Pasted image 20241222195938.png]
Das IEM versucht mit einem Neuronalen Netzwerk die Anzahl Schritte zwischen dem aktuellen und dem nächsten Zustand vorherzusagen. Die Idee ist: wenn der Wert tief ist dann kennt das Modell den Zustand schon gut respektive weiss gut wie viele Schritte benötigt werden um dahin zu kommen. Schätzt das Modell die Anzahl Schritte zum nächsten Zustand als gross ein, dann kennt es diesen Zustand noch nicht gut und es bekommt einen grösseren intrinsischen Reward wenn es diesen Zustand exploriert. Das Uncertainty-Netzwerk wird dann auf den Mean-squared errors aller gesammelten Trajectories trainiert. Das sind bei 128 trajectories $128*128/2 = 8192$ Updates, was einiges an Compute-Time hinzufügt.
Gemäss dem Paper erwarte ich, dass das Modell länger braucht zum trainieren aber dafür einen höheren durchschnittlichen Return erzielt.

![](assets/Pasted image 20241230151048.png]
Das erste Resultat zeigt, dass das Experiment mit dem Intrinsic Exploration Module viel schlechter abschneidet alls der Initialle PPO-Allgorithmus. Dennoch ist das Modell ein bisschen besser als die Random Baseline.
![](assets/Pasted image 20241230151336.png]
Wenn man den Uncertainty-Reward während dem Training verfolgt, wird schnell klar, dass dieser sehr gut maximiert wird. Da sich die Kurve aber den Maximalwert erreicht, vermute ich, dass der Agent Reward Hacking betreibt und dadurch die Exploration maximiert und gar nicht mehr wirklich versucht etwas zu lernen. Ich habe deshalb das Training mit weniger Gewichtung des Uncertainty Rewards durchgeführt. Auch weil im Paper steht, dass die Koeffiziente für jeden Task ausprobiert werden muss. Das Training hier habe ich mit einer Uncertainty-Koeffiziente von $0.1$ durchgeführt.

![](assets/Pasted image 20241230152225.png]
Ich habe die Koeffzienten von Setup zu Setup jeweils etwa durch drei dividiert: $0.1$ => $0.03$ => $0.01$ => $0.003$ ... und zuletzt noch mit einer ganz kleinen Koeffiziente versucht ($0.0000001$). Durch kleiner Uncertainty-Koeffizienten hat sich der Reward in der Evaluation schon verbessert. Das Setup mit einer Uncertainty-Koeffiziente von $0.0003$ kommt schon sehr nahe an den duchrschnittlichen Reward des initialen Setups. Allerdings sind alle Setups immernoch wesentlich schlechter als das initiale Setup und ich konnte keine Verbesserung erzielen.
![](assets/Pasted image 20241230152530.png]
Der Uncertainty Reward beginnt nach einiger Zeit auch zu sinken. Was darauf hindeutet, dass der Uncertainty Reward nicht mehr so stark eine Rolle spielt. Allerdings sink er nur um so $10\%$.

Meine Vermutung, dass IEM PPO länger brauchen wird zum Training lässt sich durch den nachfolgenden Screenshot aus Tensorboard auch bestätigen. IEM-PPO braucht fast $70\%$ länger zum Training, was vermutlich einzig dem Lernen des Uncertainty-Estimation-Network zu schulden ist.  ![](assets/Pasted image 20241230154710.png]

An diesem Punkt habe ich beschlossen keine weitere Zeit mehr in IEM PPO zu investieren. Nicht zuletzt ist es auch frustrierend, weil der Ansatz nicht vollständig beschrieben ist. Es fehlt an jeglichem Code um meine Implementation zu vergleichen oder einer genauen Beschreibung des Netzwerk/Hyperparameter. Allerdings möchte ich hier noch meine Gedanken festhalten, wie ich noch versucht hätte, das Resultat zu verbessern:
- Untersuchen des Trainings des Uncertainty-Estimation-Network: z.B. durch Hyperparameter-Suche der Lernrate oder Inspektion der Gradienten.
- Unterschiedliche Netzwerkarchitekturen ausprobieren für das Uncertainty-Estimation-Network. Die Netzwerkarchitektur ist im Paper nicht genau beschrieben. Einzig Input und Aktivierungsfunktion.
- Uncertainty-Estimation-Network direkt auf den States trainieren und nicht aus den encoded States (aus dem Encoder den auch der Actor und Critic verwenden).
- Das aktuelle Setup binned den Reward auf entweder $-1$, $0$ oder $1$. Dieses Binning könnte zum ignorieren der feinen Nuancen vom Uncertainty Reward führen, da dieser Reward eine Zahl zwischen 0 und 1 zurückgibt und nicht genau 0 oder 1.
- MaxAndSkipEnv skipped n-Frames und gibt dann das Maximum der letzten beiden Frames zurück. Dies könnte das Netzwerk eventuell verwirren.
Eine andere Möglichkeit wäre noch direkt die Autoren zu kontaktieren um mehr Informationen über die Netzwerkarchtektur zu erhalten.

Nach Besuch der Kontaktstunde habe ich herausgefunden, das solche Intrinsic Rewards vorallem bei Environments mit Sparse-Rewards etwas bringt. Das ist natürlich bei Space Invaders nicht der Fall und wahrscheinlich ein Grund, wieso dies keine Verbesserung gebrach hat. 
## 4. Frame Skip Adjustment
Bei diesem Experiment habe ich die Anzahl Frames welche geskipped werden angepasst. Im initialen Code wird nur jedes 4te Frame verwendet und es werden 4 Frames aggregiert in das Modell gefüttert. Der Agent tätigt also nur bei jedem 16ten Bild eine Action. Ich habe nun die Anzahl Frames welche ausgelassen werden einmal auf 2 reduziert und einmal 6 sowie auf 8, 16 und 32 erhöht.
![](assets/Pasted image 20241225183439.png]
Gemäss der oben stehenden Grafik hat ein Frameskipping von 2 und 32 einen negativen Einfluss auf den Return. Wobei bei 32 skipped Frames der Einfluss nur leicht negativ ist. Vorallem ist die Bandbreite von Returns bei 32 skipped Frames grösser als bei nur 4 skipped Frames. Diese grössere Bandbreite lässt sich bei allen Konfigurationen beobachten. Ausser bei 2 skipped Frames. Alle anderen Konfigurationen (ausser eben 2 und 32) haben einen positiven Einfluss auf den Return.

Ich vermute, dass bei nur 2 Skipped Frames der Unteschied zwischen den Frames nicht immer gross ist oder es überhaupt einen Unterschied gibt, da sich die Aliens nicht kontinuierlich Bewegen. Ihre Bewegung ist eher Sprunghaft, dafür tritt sie nur alle paar Frames auf. Und vermutlich eben nicht alle 2 Frames. 
Beim Skip Frame 32 Ansatz dauert es sehr lange bis der Agent wieder eine Action machen kann. So könnte es sein, dass der Agent schlicht zu wenig Aliens abschiessen kann oder viel zu spät ausweichen kann. Des Weiteren hat dieser Agent 8-mal weniger Trainingsdatenpunkte, da alle Konfigurationen die gleiche Anzahl an Timesteps haben aber unterschiedlich häufig eine Action durchführen können und deshalb auch weniger Trainingsdaten haben.

Ein weitere Interessanter Aspekt ist, dass bei 32 skipped Frames viel länger ist, obwohl der Return nur minimal kleiner ist. Dies habe ich mir nicht überlegt, ergibt aber total Sinn. Denn der Agent kann jetzt 8 mal weniger Actions ausführen in der gleichen Zeit. Dementsprechend kann der Agent auch viel weniger mal auf die Aliens schiessen. ![](assets/Pasted image 20241225190541.png]

Ein dritter und interessanter Aspekt ist, dass die Zeit pro Episode stark negativ mit den Anzahl skipped Frames korreliert. Das ist aber logisch, da bei weniger skipped Frames mehr Berechnungen ausgeführt werden, weil der Agent öfters eine Action tätigt.
![](assets/Pasted image 20241225190917.png]

## Finaler Vergleich Baseline, Initialer Ansatz und Erweiterungen
![](assets/Pasted image 20241230154756.png]
In der obenstehenden Grafik sind alle 4 Erweiterungen (jeweils nur die beste Konfiguration) sowie der initiale und Baseline Ansatz zu sehen. Hier kann man gut sehen, dass mit zwei Erweiterungen das Ergebnis im Vergleich zum initialen Ansatz verbessert werden konnte (Epsilon-Hyperparameter-Tuning und Skip-Frames). Die zwei anderen Erweiterungen (ResNet18 und IEM PPO) schnitten schlechter ab aber waren dennoch besser als die Baseline.
![](assets/Pasted image 20241230155323.png]
In der obenstehenden Grafik ist gut zusehen, dass der Reward stark mit der Episodenllänge korreliert. Je mehr Steps ein Agent im Environment ausführt, desto mehr Reward erhält er im Durchschnitt und umgekehrt.
![](assets/Pasted image 20241230155515.png]
Die Durchschnittliche Laufzeit pro Episode in der Evaluation zeichnet aber ein anderes Bild als eine Korrelation mit dem Reward. Hier wird deutlich, dass die verschiedenen Erweiterungen einen deulichen Einfluss auf nötigen Ressourcen (Rechenleistung) haben. Während das Hyperparameter-Tuning genaus gleich viel Ressourcen braucht wie der initiale Ansatz braucht der deutliche grössere und komplexere ResNet18 Agent viel mehr Rechenleistung bei kleinereen Episoden. Der IEM-PPO Algorithmus brauch ein bisschen weniger Zeit, allerdings sind dessen Episoden auch kürzer (siehe Grafik oben). Dies ergibt Sinn, da das Uncertainty-Estimation-Netzwerk nicht verwendet wird in der Evaluation. Hingegen der Skip-Frame Ansatz verwendet deutlich weniger Ressourcen, da der Agent für die gleiche Anzahl an Timesteps weniger Actions berechnen muss als der initiale Ansatz.  

![](assets/Pasted image 20241225191756.png]
Interessant: Skip-Frames 16 ist fast so schnell wie Baseline bei der Evaluation. Das deutet darauf hin, das dort die meiste Rechenleistung für die Simulation des Environment aufgewendet wird und nur äusserst wenig für die Entscheidugnen des Agenten.

| Ansatz        | Dauer    |
| ------------- | -------- |
| Initial       | 30min    |
| Epsilon       | 30min    |
| IEM-PPO       | 50min    |
| ResNet        | 45min    |
| Skip Frame 2  | 23min    |
| Skip Frame 6  | 34       |
| Skip Frame 8  | 50       |
| Skip Frame 16 | 1h 34min |
| Skip Frame 32 | 3h 12min |
Im Training lassen sich ein bisschen andere Erkenntnisse feststellen bezüglich der Trainingsdauer. Der Initiale-Ansatz und das Hyperparameter-Tuning brauchen gleiche lange, da sie die gleichen Operation gleich häufig ausführen. Nur eine Koeffiziente verändert sich. IEM-PPO braucht wesentlich länger, da es ein weiteres Netzwerk aktualisieren und für den Reward jeweils auch auswerten muss. ResNe braucht auch länger als der initiale Ansatz, da die Berechnungen der Action durch den Agent durch das wesentlich grössere Netzwerk komplizierzter werden und dem entsprechend mehr Rechenleistung/Zeit brauchen.

Völlig anderst sieht es bei den Skip-Frame-Konfigurationen aus. Hier werden unterschiedlich viele Frames pro Zeitschritt simuliert. Alle Konfigurationen haben aber genau gleich viel Zeitschritte. Somit wächst hier die Trainingsdauer mit den Anzahl skipped Frames.  
## Fazit
Mit zwei Erweiterungen konnte ich die Perfomance des initialen Ansatzes (PPO) verbessern wobei bei einem auch die Varianz des Returns grösser wurde (Frameskip: 16) und nur beim Anderen ($\epsilon = 0.2$) die Varianz auch kleiner wurde. Dass die Erweiterung $\epsilon = 0.2$ besser abschnitt verwundert mich nicht, da ich dies überall in der Literatur gelesen habe. Was ich aber weniger erwartet habe, ist die Verbesserung durch mehr skipped Frames, da der Agent dann weniger Möglichkeiten hat mit dem Spaceship die Aliens abzuschiessen.

Die Vorlagen (`ppo_clean_rl.py`) hat zwar einen einfacheren Start ermöglicht als wenn ich von Scratch angefangen hätte. Allerdings hatte ich eine sehr viel neuere Version von `gymnasium` (`1.0.0`) welche mit erheblichen Änderungen an der API einherging. Dementsprechend musste ich einiges debuggen und abändern bis der Code lief. Die grösste Änderung war wohl, wie das Environment signalisiert, wenn eine Episode terminiert hat. im `infos` Dict gibt es jetzt kein Keyword `final_info` mehr.

Am meisten Kopfweh hat mir die Implementierung von IEM-PPO bereitet. Das Paper hat keinen Referenz-Code und die optimalen Hyperparameter sind nicht dokumentiert. Dennoch habe ich viel gelernt in dieser Mini-Challenge und freude am Ausprobieren von diversen Konfigurationen.
## Future Work
Aufgrund der knappen Zeit für dieses Modul konnte ich nicht alle Verbesserungen umsetzen welche ich vor hatte und die Evaluation hat auch noch Luft nach oben. Spezifisch hätte ich gerne foglende Dinge verbessert um meine Evaluation und Experimente robuster zu machen.
- **Mehr Zeitrschritte:** Ich hätte gerne noch länger als nur eine Million Zeitschritte trainiert. Wenn ich mir die Losskurven und die Returnkurven ansehe, dann sehe ich da noch kein Plateau. 
- **Kreuzvalidierung für Unsicherheitsabschätzung:** Des Weiteren hätte ich gerne noch eine Kreuzvalidierung eingebaut um die Unsicherheiten bei den berechneten Kennzahlen aus der Evaluation abzuschätzen (da RL-Algorithmen - wie andere ML-Algorithmen - ja instabil sein können). Bei manchen Ergebnissen ist mir sehr unklar ob sie jetzt nur durch Zufall minimal besser/schlechter sind als der initiale Ansatz.
- **Weiter Hyperparameter testen:** Es ist sicherlich auch noch interessant, mit weiteren Hyperparameter zu experimentieren, aber dazu hat leider auch die Zeit gefehlt.
- **Qualitative Evaluation des Verhaltens:** Des Weiteren hatte ich nicht genug Zeit für eine ausführliche qualitative Evaluation des Agent indem ich mir Videos des Verhaltens des Agenten genauer unter die Lupe genommen hätte. Das würde helfen Muster, Verhaltensweisen und Fehler (z.B. Reward Hacking) des Agenten zu erkennen und darauf aufbauend dann den Algorithmus verbessern.
- **Auf mehreren Atari Umgebungen gleichzeitig trainieren:** Auch hätte mich interessiert, wie ob sich der Agent verbessert, wenn er nicht nur auf Space Invaders trainiert wird sondern gleichzeitig auch noch auf anderen Umgebungen der Atari Games. Damit bekommt der Agent hoffentlich ein besseres und allgemeineres Verständnis der Spiele.
- **Gleiche Anzahl an Trainingsdatenpunkte bei Skip Frame:** Um zu Überprüfen, ob die kleinere Anzahl von Samples (gerade bei 32 skipped Frames) auch einen Einfluss auf die Performance hat, oder ob dies alles attributiert werden kann an den Fakt, dass der Agent viel (8-mal) weniger Actions vornehmen kann als im initialen Ansatz, würde ich gerne das Training mit Skipped Frames so durchführen, dass alle Setups gleich viel Trainingsdatenpunkte haben. Mit der Konsequenz, dass nicht alle Setups gleich viel Timesteps haben.