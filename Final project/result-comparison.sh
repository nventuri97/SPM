#!/bin/zsh

# Compilazione del codice
g++ -I ~/fastflow -std=c++20 -Wall -O3 -ffast-math -DNDEBUG -o FFUTWavefront FFUTWavefront.cpp -pthread
g++ -I ../include -std=c++20 -Wall -O3 -DNDEBUG -o SequentialUTWavefront SequentialUTWavefront.cpp

# Esecuzione 5 volte
for i in {1..5}; do
    echo "Esecuzione $i"
    
    # Esegui i programmi e cattura l'output
    rF=$(./FFUTWavefront 5 4)
    rS=$(./SequentialUTWavefront 5)
    
    # Stampa i risultati per debug
    echo "Risultato FFUTWavefront:"
    echo "$rF"
    echo "Risultato SequentialUTWavefront:"
    echo "$rS"

    # Confronta i risultati
    if [ "$rF" = "$rS" ]; then
        echo "I risultati sono uguali."
    else
        echo "I risultati sono diversi."
    fi
    echo
done
