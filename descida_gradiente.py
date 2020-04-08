import numpy as np


class DescidaGradienteRosenbrock():

    TOLERANCIA_MAXIMA = 1 * (10 ** -4)
    NUM_MAX_PASSOS = 20000
    VALOR_INICIAL = np.array([0.0, 0.0, 0.0], dtype=float)


    def __init__(self, learning_rate):
        self.LEARNING_RATE = learning_rate


    def executar(self):
        self.passo = 1
        self.valor_atual = None
        self.valor_anterior = None
        self.tolerancia = 1000000
        self.ponto = self.VALOR_INICIAL

        self.calc_valor_atual()
        print(self.ponto, self.valor_atual)

        while self.tolerancia >= self.TOLERANCIA_MAXIMA and self.passo <= self.NUM_MAX_PASSOS:
            # Calculando o valor do gradiente
            self.calc_gradiente()

            # Calculando o novo ponto em função do gradiente e da taxa de convergência
            self.ponto = self.ponto - (self.LEARNING_RATE * self.gradiente)
            self.valor_anterior = self.valor_atual
            self.calc_valor_atual()

            # Calculando nova tolerância
            self.calc_tolerancia()

            # Imprimindo resultado desse passo
            print("(%d) tolerancia=%f anterior=%.16f valor=%.16f" % (self.passo, self.tolerancia, self.valor_anterior, self.valor_atual))

            self.passo += 1


    def calc_gradiente(self):
        x1 = self.ponto[0]
        x2 = self.ponto[1]
        x3 = self.ponto[2]

        # Derivadas obtidas pelo WolframAlpha
        # Gradiente na direção X1
        grad_x1 = 2 * ((200 * (x1 ** 3)) - (200 * x1 * x2) + x1 -1)

        # Gradiente na direção X2
        grad_x2 = (-200 * (x1 ** 2)) + (400 * (x2 ** 3)) + (x2 * (202 - (400 * x3))) - 2

        # Gradiente na direção X3
        grad_x3 = 200 * (x3 - (x2 ** 2))

        self.gradiente = np.array([grad_x1, grad_x2, grad_x3], dtype=float)


    def calc_tolerancia(self):
        n1 = abs(self.valor_atual - self.valor_anterior)
        n2 = abs(self.valor_anterior)

        self.tolerancia =  n1 / n2 if n1 and n2 else 0


    def calc_valor_atual(self):
        self.valor_atual = self.func_rosenbrock(self.ponto[0], self.ponto[1], self.ponto[2])


    def func_rosenbrock(self, x1, x2, x3):
        resp = 100 * ((x2 - (x1 ** 2)) ** 2)
        resp += (1 - x1) ** 2
        resp += 100 * ((x3 - (x2 ** 2)) ** 2)
        resp += (1 - x2) ** 2

        return resp



# CÓDIGO PRINCIPAL
descida = DescidaGradienteRosenbrock(1 * (10 ** -4))
descida.executar()
